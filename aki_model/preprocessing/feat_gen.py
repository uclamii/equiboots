################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import os
import sys

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, "aki_model")

from functions import mlflow_dumpArtifact, mlflow_loadArtifact

from constants import (
    exp_artifact_name,
    outcomes,
    preproc_run_name,
    target,
    var_index,
    categorical_cols,
    sensitive_cols,
    treatment_col,
)

################################################################################
################ Model Preprocessing and Feature Engineering ###################
################################################################################

################################################################################
################ Step 2: Define Typer Application ##############################
################################################################################

app = typer.Typer()

################################################################################
################ Step 3: Define Main Function ##################################
################################################################################


@app.command()
def main(
    input_data_file: str = "./data/processed/df_sans_zero.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
    drop_treatment: bool = False,
    track: bool = True,
):
    """
    Processes the input data file and generates feature space X and target
    variable y.

    y is the single binary column named by `target` in constants.py. The rest
    of the outcome family declared in `outcomes` is removed from X, since the
    composite is reconstructable from its three components exactly and the
    time_to_* columns encode the same information continuously.

    No per-column missingness indicators are produced. Missingness on this
    deposit runs from 0.03% to 1.71% across nineteen lab and vital columns, and
    those nineteen collapse to fifteen distinct patterns once the exact
    duplicates are removed. Pooled across all of them, 119 patients have any
    value missing, at a 0.261 event rate against a 0.211 base: 31 events where
    25 were expected, which is well inside noise. Nineteen near-constant columns
    is a real cost in exchange for that. The row-level rate carried over from
    preprocess.py holds the same signal in one column, and the NA values
    themselves survive in X for an imputer fitted inside the CV folds.

    Args:
        input_data_file (str): Path to the input parquet file.
        stage (str): 'training' or 'inference'.
        data_path (str): Directory for X.parquet and y.parquet.
        drop_treatment (bool): Remove the randomization arm from X. Off by
            default, which gives a prognostic model conditional on assignment.
        track (bool): Log the feature list to MLflow.
    """
    if stage not in ("training", "inference"):
        raise typer.BadParameter("stage must be 'training' or 'inference'")

    os.makedirs(data_path, exist_ok=True)

    ############################################################################
    ################ Step 4: Load Input Data ###################################
    ############################################################################

    df = pd.read_parquet(input_data_file)

    # Set index if not already set
    if df.index.name != var_index:
        try:
            df.set_index(var_index, inplace=True)
            print(f"Index set to '{var_index}'.")
        except KeyError:
            print(
                f"Warning: '{var_index}' not found in columns - "
                "proceeding with default integer index."
            )
    else:
        print(f"Index '{var_index}' already set - skipping.")

    print(df)

    print("-" * 80)
    print(f"# of DataFrame Columns: {df.shape[1]}")

    ############################################################################
    ################ Step 5: Training Stage ####################################
    ############################################################################

    if stage == "training":

        if target not in df.columns:
            raise ValueError(
                f"target '{target}' absent from {input_data_file}. Check that "
                "preprocessing.py carried the outcome family through PROTECTED."
            )

        ############### Separate features (X) and target (y) ###################
        # Every member of `outcomes` leaves X, not just the label. The composite
        # equals (aki_progression14 | death14 | dialysis14) on all 6,030 rows,
        # so leaving any component behind hands the model the answer, and the
        # time_to_* columns do the same in continuous form.
        ########################################################################

        drop_cols = [c for c in outcomes if c in df.columns]

        X = df.drop(columns=drop_cols).copy()
        y = df[target].copy()  # single column; this is a binary task

        ############### Treatment assignment ###################################
        # `alert` is a design variable rather than a risk factor. Retained by
        # default so the model is prognostic conditional on assignment; drop it
        # for a pure baseline-risk model.
        ########################################################################

        if drop_treatment and treatment_col in X.columns:
            X = X.drop(columns=[treatment_col])
            print(f"\nDropped treatment column '{treatment_col}' from X.")

        ## Log first five rows of features and targets
        print(f"\n{'=' * 80}\nX\n{'=' * 80}\n{X.head()}")
        print(f"\n{'=' * 80}\ny ({target})\n{'=' * 80}\n{y.head()}")

        ############### Guard against outcome bleed-through ####################
        residual = sorted(set(outcomes) & set(X.columns))
        if residual:
            raise ValueError(f"Outcome columns still present in X: {residual}")

        ## Display class balance
        counts = y.value_counts().sort_index()
        print(f"\nBreakdown of y:\n{counts}")
        print(f"\nPositive rate: {y.mean():.4f}")
        print(
            f"Imbalance ratio (neg:pos): {counts.get(0, 0) / max(counts.get(1, 1), 1):.2f}:1"
        )

        ############### Report structure available downstream ##################
        print(
            f"\nCategorical columns in X: {[c for c in categorical_cols if c in X.columns]}"
        )
        print(
            f"Sensitive attributes in X: {[c for c in sensitive_cols if c in X.columns]}"
        )
        print(f"Treatment column in X: {treatment_col in X.columns}")

        X_columns_list = X.columns.to_list()

    ############################################################################
    ################ Step 6: Inference Stage Load X_columns list ###############
    ############################################################################

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Features List From `feat_gen.py`
        ########################################################################
        # During training we stored `X_columns_list`. Reloading it is what keeps
        # inference on the same footing as training: same columns, same order.
        ########################################################################

        X_columns_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,  # Use the same run_name as training
            obj_name="X_columns_list",
        )

        absent = [c for c in X_columns_list if c not in df.columns]
        if absent:
            raise ValueError(
                f"Features seen at training are absent at inference: {absent}"
            )

        X = df[X_columns_list].copy()

        # y is optional at inference; produced only if the label came along.
        y = df[target].copy() if target in df.columns else None

    ############################################################################
    ################ Step 7: Store Final List of Features for Production #######
    ############################################################################

    if stage == "training" and track:
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,  # Consistent run_name for all artifacts
            obj_name="X_columns_list",
            obj=X_columns_list,
        )

    # Pin column order to the training contract.
    X = X[X_columns_list]

    print(f"\nShape of X: {X.shape} \n")
    if stage == "inference":
        print(
            "\033[33mNumber of rows may vary due to row filtering in "
            "`preprocessing.py`\033[0m"
        )
    print("-" * 80)
    print(f"\nFeature Space\n{X.head()}\n")

    ############################################################################
    ################ Step 8: Generate Target Variable for Training #############
    ############################################################################

    if stage == "training":
        # Single binary target from constants.py, written as a one-column frame
        # so the parquet round-trip keeps the column name.
        y = y.astype(int).to_frame(name=target)

        if not X.index.equals(y.index):
            raise ValueError("X and y indices diverged; refusing to write.")

        y.to_parquet(os.path.join(data_path, "y.parquet"))

    elif y is not None:
        y = y.astype(int).to_frame(name=target)
        y.to_parquet(os.path.join(data_path, "y_inference.parquet"))

    ############################################################################
    ################ Step 9: Save Processed Feature Space ######################
    ############################################################################

    # Save the feature space (X) and target variables (y) to parquet files
    X.to_parquet(os.path.join(data_path, "X.parquet"))

    ############################################################################
    ################ Step 10: Summary ##########################################
    ############################################################################

    n_num = X.select_dtypes(include=np.number).shape[1]
    n_cat = X.select_dtypes(include=["category", "object"]).shape[1]

    print("-" * 80)
    print(f"X: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"  numeric / categorical: {n_num} / {n_cat}")
    if stage == "training":
        print(f"y: {target}, positive rate {y[target].mean():.4f}")
    print(
        f"Residual NA in X: {int(X.isna().sum().sum())} cells across "
        f"{int(X.isna().any().sum())} columns (impute inside your CV folds)"
    )
    print()


################################################################################

if __name__ == "__main__":
    app()
