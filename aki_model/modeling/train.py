import sys
import pandas as pd
from pathlib import Path
import typer
from loguru import logger

from sklearn.feature_selection import RFE
from model_tuner import Model

################################################################################
# Step 1. Import Configurations and Constants
################################################################################

sys.path.insert(0, "aki_model")

from config import (
    PROCESSED_DATA_DIR,
    model_definitions,
    rstate,
    pipelines,
    numerical_cols,
    categorical_cols,
)
from functions import (
    clean_feature_selection_params,
    mlflow_log_parameters_model,
    adjust_preprocessing_pipeline,
    mlflow_load_model,
)

app = typer.Typer()

################################################################################
# Step 2. Define CLI Arguments with Default Values
################################################################################


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ---
    model_type: str = "lr",  # lr # rf # xgb # cat
    pipeline_type: str = "orig",
    outcome: str = "Label_ESKD_2_years",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y.parquet",
    scoring: str = "average_precision",  # "roc_auc",
    pretrained: int = 0,
    cat_native: int = 1,  # use native catboost categorical handling if 1
    # -----------------------------------------
):

    ################################################################################
    # Step 3. Load Feature and Label Datasets
    ################################################################################

    X = pd.read_parquet(features_path)  # read in X

    # Convert categorical columns to strings
    # X[categorical_cols] = X[categorical_cols].astype(str)

    # uniform strings for the categorical branch; NaN stays real for the imputer
    for c in categorical_cols:
        X[c] = X[c].astype(object).map(lambda v: v if pd.isna(v) else str(v))

    y = pd.read_parquet(labels_path)  # read in y
    y = y[outcome].squeeze()  # coerce into a series

    ############################################################################
    # Step 3a. Group-level split preparation (only use if group split)
    ############################################################################

    stratify_df = X[
        [
            "white",
            "asian",
            "native",
            "islanders",
            "Hispanic",
            "MultiRacial",
            "Female",
        ]
    ]

    race = pd.Series("other", index=stratify_df.index)
    race[stratify_df["asian"].astype(bool)] = "asian"
    race[stratify_df["white"].astype(bool)] = "white"
    race[stratify_df["Hispanic"].astype(bool)] = "hispanic"

    combo = race + "_" + stratify_df["Female"].astype(str)

    ################################################################################
    # Step 4. Retrieve Model and Pipeline Configurations
    ################################################################################

    clc = model_definitions[model_type]["clc"]
    estimator_name = model_definitions[model_type]["estimator_name"]
    pipeline_steps = pipelines[pipeline_type]["pipeline"]
    sampler = pipelines[pipeline_type]["sampler"]
    feature_selection = pipelines[pipeline_type]["feature_selection"]

    # Set the parameters
    tuned_parameters = model_definitions[model_type]["tuned_parameters"]
    randomized_grid = model_definitions[model_type]["randomized_grid"]
    n_iter = model_definitions[model_type]["n_iter"]
    early_stop = model_definitions[model_type]["early"]

    print("Sampler", sampler)

    ################################################################################
    # Step 5. Clean up pipeline
    # Step 5a. Clean up tuned_parameters by removing feature selection keys if
    # RFE isn't in the pipeline
    ################################################################################
    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Step 5b. Adjust preproc. pipe. to skip imputer and scaler for 'rf', 'xgb', 'cat'

    pipeline_steps, is_native = adjust_preprocessing_pipeline(
        model_type,
        pipeline_steps,
        numerical_cols,
        categorical_cols,
        sampler=sampler,
        native_cat=cat_native,
    )

    # Always rebuild the preprocessor with updated columns
    for i, (name, step) in enumerate(pipeline_steps):
        if "Preprocessor" in name:
            step.transformers = [
                ("num", step.transformers[0][1], numerical_cols),
                ("cat", step.transformers[1][1], categorical_cols),
            ]
            break

    ############################################################################
    # Step 5c. Determine RFE status and CatBoost fit_params
    ############################################################################
    has_rfe = any(isinstance(step[1], RFE) for step in pipeline_steps)
    cat_feature_indices = list(
        range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))
    )

    cat_tag = "_native" if is_native else ""

    # - CatBoost (native): pass cat_features indices for native categorical handling
    # - XGBoost (native): enable_categorical=True already set in config; no
    #   fit_params needed
    # - OHE path (RFE, SMOTE, or --cat-native 0): categoricals are one-hot
    #   encoded to floats, so cat_features indices would be invalid
    if model_type == "cat" and is_native:
        fit_params = {"cat__cat_features": cat_feature_indices}
    else:
        fit_params = {}

    ################################################################################
    # Step 6. Printing Outcome
    ################################################################################

    print()
    print(f"Outcome:")
    print("-" * 60)
    print()
    print("=" * 60)
    print(f"{outcome}")
    print("=" * 60)

    ################################################################################
    # Step 7. Define and Initialize the Model Pipeline
    ################################################################################

    logger.info(f"Training {estimator_name} for {outcome} ...")

    if pretrained:

        print("Loading Pretrained Model...")
        model = mlflow_load_model(
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}{cat_tag}_training",
            model_name=f"{estimator_name}_{outcome}",
        )

    else:
        model = Model(
            pipeline_steps=pipeline_steps,
            name=estimator_name,
            model_type="classification",
            estimator_name=estimator_name,
            calibrate=True,
            estimator=clc,
            kfold=False,
            grid=tuned_parameters,
            n_jobs=5,
            randomized_grid=randomized_grid,
            n_iter=n_iter,
            scoring=[scoring],
            random_state=rstate,
            stratify_cols=combo.to_frame("strat"),  # CH: used_in_6month (uncomment)
            stratify_y=True,
            boost_early=early_stop,
            imbalance_sampler=sampler,
            feature_selection=feature_selection,
            # groups=groups, #only use if doing group-splits
        )

        ################################################################################
        # Step 8. Perform Hyperparameter Tuning
        ################################################################################

        model.grid_search_param_tuning(
            X,
            y,
            f1_beta_tune=True,
            betas=[1],
            fit_params=fit_params,
        )

        ################################################################################
        # Step 9. Extract Training, Validation, and Test Splits
        ################################################################################

    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test = model.get_test_data(X, y)

    print(f"Train/Valid/Test sizes:")

    print(X_train.shape, X_valid.shape, X_test.shape)

    print(
        f"Total Train_Val_Test size: {X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]}"
    )

    ################################################################################
    # Step 10. Train the Model
    ################################################################################

    # Boosting algorithms like XGBoost and CatBoost benefit from validation data
    # during training to optimize early stopping and prevent overfitting.
    # Hence, we explicitly provide the validation dataset in the `fit` method
    # for these models. For other models, validation data is not required at this
    # stage.

    if not pretrained:
        if model_type == "cat" and not has_rfe:
            # CatBoost native categorical: validation_data safe, cat_features passed
            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                score=scoring,
                fit_params=fit_params,
            )
        elif model_type == "xgb" and not has_rfe:
            # XGBoost native categorical: category dtype preserved via
            # set_output("pandas") on ColumnTransformer; consistent codes
            # ensured by combined_cats above
            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                score=scoring,
            )
        elif model_type in {"xgb", "cat"}:
            # RFE path: OrdinalEncoder used, validation_data safe
            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                score=scoring,
                fit_params=fit_params,
            )
        else:
            model.fit(
                X_train,
                y_train,
                score=scoring,
            )

    ################################################################################
    # Step 11. Calibrate the Model If Necessary
    ################################################################################

    # ## If we need to update isotonic method
    # model.calibration_method = "isotonic"

    if model.calibrate:
        if model_type in {"xgb", "cat"}:
            model.calibrateModel(
                X,
                y,
                f1_beta_tune=True,
                fit_params=fit_params,
            )
        else:
            model.calibrateModel(
                X,
                y,
                score=scoring,
                f1_beta_tune=True,
            )

    ################################################################################
    # Step 12. See Results in Terminal and Store Model in MLFlow
    ################################################################################

    # see the results printed to the terminal for reference
    print("\nValidation Performance:\n")
    model.return_metrics(
        X=X_valid,
        y=y_valid,
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )
    print("\nTest Set Performance:\n")
    model.return_metrics(
        X=X_test,
        y=y_test,
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )
    if pretrained:
        mlflow_log_parameters_model(
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}{cat_tag}_training",
            model_name=f"{estimator_name}_{outcome}",
            model=model,
        )

    else:
        mlflow_log_parameters_model(
            model_type=model_type,
            n_iter=n_iter,
            kfold=False,
            outcome=outcome,
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}{cat_tag}_training",
            model_name=f"{estimator_name}_{outcome}",
            model=model,
            hyperparam_dict=model.best_params_per_score[scoring],
        )

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
