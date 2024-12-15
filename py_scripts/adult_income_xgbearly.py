### Adult Income Dataset

######################### Import Requisite Libraries ###########################

import model_tuner

print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()


from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import os
import sys
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from eda_toolkit import add_ids
from model_tuner import Model

if __name__ == "__main__":

    ################################# Set Data Path ################################
    data_path = "./public_data"

    ############################### Load The Dataset ###############################

    ### fetch dataset
    adult = fetch_ucirepo(id=2)

    ##################### Define the Feature Space and Outcome #####################

    # fetch dataset
    adult = fetch_ucirepo(id=2)

    adult = adult.data.features.join(adult.data.targets, how="inner")

    adult = add_ids(
        df=adult,
        id_colname="Adult_ID",
        num_digits=9,
        seed=222,
    ).set_index(
        "Adult_ID",
    )

    ########################## Map Income to Booleans ##############################
    ################### and output adult income dataset to .csv ####################

    adult.loc[:, "income"] = adult["income"].str.rstrip(".")  # Remove trailing periods
    adult["income"] = adult["income"].map({"<=50K": 0, ">50K": 1})

    adult_subset = adult[["race", "sex", "income"]]

    adult.to_csv(os.path.join(data_path, "adult.csv"))

    ######################## data (as pandas dataframes) ###########################
    X = adult[[col for col in adult.columns if not "income" in col]]
    y = adult[["income"]]

    print("-" * 80)
    print("X")
    print("-" * 80)

    print(X.head())  # inspect first 5 rows of X

    print("-" * 80)
    print("y = Outcome = Income")
    print("-" * 80)

    print(f"\n{y.head()}")  # inspect first 5 rows of y

    print(f"\n Income Value Counts: \n")

    # Check the updated value counts
    print(y["income"].value_counts())

    ################### Parse Categorical and Numerical Features ###################
    # >2 categories
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "race",
    ]

    ## continuous or binary
    numerical_features = X.select_dtypes(np.number).columns.to_list()

    ################### Create an instance of the XGBRegressor #####################

    xgb_name = "xgb"
    xgb = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        random_state=222,
    )

    ##################### Define hyperparameters for XGBoost #######################

    xgbearly = True
    tuned_parameters_xgb = {
        f"{xgb_name}__max_depth": [3, 10, 20, 200, 500],
        f"{xgb_name}__learning_rate": [1e-4],
        f"{xgb_name}__n_estimators": [1000],
        f"{xgb_name}__early_stopping_rounds": [100],
        f"{xgb_name}__verbose": [0],
        f"{xgb_name}__eval_metric": ["logloss"],
    }

    xgb_definition = {
        "clc": xgb,
        "estimator_name": xgb_name,
        "tuned_parameters": tuned_parameters_xgb,
        "randomized_grid": False,
        "n_iter": 5,
        "early": xgbearly,
    }

    model_definitions = {
        xgb_name: xgb_definition,
    }

    # Define transformers for different column types
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Create the ColumnTransformer with passthrough
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    model_type = "xgb"
    clc = xgb_definition["clc"]
    estimator_name = xgb_definition["estimator_name"]

    tuned_parameters = xgb_definition["tuned_parameters"]
    n_iter = xgb_definition["n_iter"]
    rand_grid = xgb_definition["randomized_grid"]
    early_stop = xgb_definition["early"]
    kfold = False
    calibrate = True

    ##################### Initialize and configure the Model #######################

    model_xgb = Model(
        name=f"AIDS_Clinical_{model_type}",
        estimator_name=estimator_name,
        calibrate=calibrate,
        estimator=clc,
        model_type="classification",
        kfold=kfold,
        pipeline_steps=[("ColumnTransformer", preprocessor)],
        stratify_y=True,
        stratify_cols=["race", "sex"],
        grid=tuned_parameters,
        randomized_grid=rand_grid,
        boost_early=early_stop,
        scoring=["roc_auc"],
        random_state=222,
        n_jobs=2,
    )

    ####################### Grid Search Parameter Tuning ###########################

    model_xgb.grid_search_param_tuning(X, y, f1_beta_tune=True)

    ####################### Extract Train, Val, Test Splits ########################

    X_train, y_train = model_xgb.get_train_data(X, y)
    X_test, y_test = model_xgb.get_test_data(X, y)
    X_valid, y_valid = model_xgb.get_valid_data(X, y)

    ################################# Fit The Model ################################

    model_xgb.fit(X_train, y_train, validation_data=[X_valid, y_valid])

    ######################### Return Metrics (Optional) ############################

    print("Validation Metrics")
    model_xgb.return_metrics(X_valid, y_valid, optimal_threshold=True)

    print("Test Metrics")
    model_xgb.return_metrics(X_test, y_test, optimal_threshold=True)

    ##################### Extract Predicted Probabilities ##########################

    y_prob = model_xgb.predict_proba(X_test)
    y_prob = pd.DataFrame(
        y_prob,
        index=X_test.index,
        columns=["prob_class_0", "prob_class_1"],
    )

    ## Save out `y_prob` to csv in `public_data` path
    y_prob.to_csv(os.path.join(data_path, "y_prob.csv"))

    ########################### Extract Predictions ################################

    y_pred = model_xgb.predict(X_test, optimal_threshold=True)

    y_pred = pd.DataFrame(y_pred, index=X_test.index, columns=["predicted"])

    adult_predictions = adult_subset.join(y_pred, on="Adult_ID", how="inner")

    ## Save out `adult_predictions` to csv in `public_data` path
    adult_predictions.to_csv(os.path.join(data_path, "adult_predictions.csv"))
