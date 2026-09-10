import os
import sys

import numpy as np
import pandas as pd
import typer
from rich.console import Console

sys.path.insert(0, "aki_model")

from functions import (
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    should_drop,
)

################################################################################
######################### Import Requisite Libraries ###########################


# import pickling scripts
from model_tuner.pickleObjects import dumpObjects

################################################################################

from constants import (
    AGE_COL,
    AGE_FLAG_COL,
    AGE_MASK_FILL,
    BASELINE_CREAT,
    CREAT_AT_RAND,
    MINCREAT_48,
    PROTECTED,
    exp_artifact_name,
    miss_col_thresh,
    percent_miss,
    preproc_run_name,
    var_index,
)

from config import categorical_cols

app = typer.Typer()
console = Console()


@app.command()
def main(
    input_data_file: str = str("./data/raw/ELAIA-1_deidentified_data_10-6-2020.csv"),
    output_data_file: str = str("./data/processed/df_sans_zero.parquet"),
    stage: str = "training",
    data_path: str = str("./data/processed/"),
    track: bool = True,
):
    """
    Preprocess the ELAIA-1 deposit up to, but not including, the X/y split.

    X and y are established downstream in feat_gen.py. This script is
    responsible for the decisions that must be identical between training and
    inference: which columns exist at all, what type they are, and how the
    de-identification artifacts in the deposit are repaired.

    Args:
        input_data_file (str): Path to the input csv or parquet file.
        output_data_file (str): Path to save the processed parquet file.
        stage (str): Processing stage ('training' or 'inference').
        data_path (str): Directory for pickled column lists.
        track (bool): Log column lists to MLflow. Set False for a dry run.
    """
    os.makedirs(data_path, exist_ok=True)

    ############################################################################
    # Step 1. Read the input data file
    ############################################################################

    df = pd.read_csv(input_data_file)

    print(df.head())
    print("=" * 90)

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

    print(df.head())
    print("=" * 90)
    print(
        "\nThe original df has",
        df.shape[0],
        "rows and",
        df.shape[1],
        "columns.\n",
    )
    print("There are", df.index.nunique(), "unique patients in the df.\n")

    ############################################################################
    # STEP 2: Drop redundant and leaking columns
    #
    # The drop rules live in constants.py: PROTECTED, LEAKAGE, REFERENCE_LEVELS,
    # PREFIXES, SUFFIXES; the reasoning for every entry is documented there. The
    # should_drop() function that applies them is in functions.py.
    #
    # reasons is kept rather than discarded so the drop list is auditable.
    #
    # ELAIA-1 is a randomized trial, so the boundary being enforced here is
    # randomization: a column survives only if its value was fixed at or before
    # that moment. The outcome family is PROTECTED and passes through untouched
    # for feat_gen.py to resolve into a label.
    ############################################################################

    reasons = {c: should_drop(c) for c in df.columns}
    df = df.drop(columns=[c for c, r in reasons.items() if r])

    ############################################################################
    # STEP 3: Report what was removed
    #
    # value_counts() over the reasons gives the one-slide summary for review.
    ############################################################################

    console.rule("[bold]Drop rules")
    print(f"columns in:  {len(reasons)}")
    print(f"columns out: {df.shape[1]}\n")
    print("Columns dropped by reason:")
    print(pd.Series([r for r in reasons.values() if r]).value_counts())
    print()
    print("Dropped columns:")
    print(sorted(c for c, r in reasons.items() if r))
    print()
    print(f"Outcome family carried forward: {sorted(PROTECTED & set(df.columns))}")

    if track:
        dropped_cols = {c: r for c, r in reasons.items() if r}
        dumpObjects(dropped_cols, os.path.join(data_path, "dropped_cols.pkl"))
        if stage == "training":
            mlflow_dumpArtifact(
                experiment_name=exp_artifact_name,
                run_name=preproc_run_name,
                obj_name="dropped_cols",
                obj=dropped_cols,
            )

    ############## Deterministic row order #####################################
    df = df.sort_index()

    if stage == "training":

        df_string = df.select_dtypes(include=["object", "category", "string", "bool"])
        dtype_counts = df_string.dtypes.astype(str).value_counts()

        print()
        print(
            "The following columns have strings or booleans and may need to be "
            "removed from modeling and/or otherwise transformed. This list is "
            "stored as an artifact in MLflow for future reference if necessary "
            f"for retrieval at a later time. \n \n"
            f"There are {df_string.shape[1]} non-numeric columns:\n \n"
            + "".join(f"  {dt:<12} {n:>5}\n" for dt, n in dtype_counts.items())
            + f"\n{df_string.columns.to_list()}. \n "
        )

        ########################################################################
        # Step 3a. String Columns Handling
        ########################################################################
        # String columns are identified and should be removed before modeling
        # because machine learning models typically require numerical inputs.
        # Keeping string columns in the dataset may lead to errors or
        # unintended behavior unless explicitly encoded.
        #
        # The ELAIA-1 deposit arrives fully numeric, so this list is expected to
        # be empty. It is still logged, because an empty list is evidence and a
        # silently skipped step is not.
        ########################################################################

        string_cols_list = df_string.columns.to_list()

        ########################################################################
        # Step 3b. Save and Log String Column List
        ########################################################################

        dumpObjects(
            string_cols_list,
            os.path.join(data_path, "string_cols_list.pkl"),
        )

        if track:
            mlflow_dumpArtifact(
                experiment_name=exp_artifact_name,
                run_name=preproc_run_name,
                obj_name="string_cols_list",
                obj=string_cols_list,
            )

    ############################################################################
    # Step 3c. Transform Boolean Columns to Integer for Modeling
    ############################################################################

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    ############################################################################
    ###################### Re-engineering Selected Features ####################
    ############################################################################

    ############################################################################
    # Step 4a. Repair the HIPAA age mask
    #
    # age is blank for exactly the patients flagged age_over_90 == 1, and for no
    # one else. That is Safe Harbor censoring of ages 90 and above, not missing
    # data, and it must be repaired before the missingness machinery downstream
    # sees an 8.4% gap and treats it as a quality problem.
    #
    # The assertion is deliberate. If a future deposit revision breaks the
    # one-to-one correspondence, the run should stop rather than quietly fill
    # genuine missing ages with 90.
    ############################################################################

    console.rule("[bold]Feature re-engineering")

    if AGE_COL in df.columns and AGE_FLAG_COL in df.columns:
        masked = df[AGE_COL].isna()
        flagged = df[AGE_FLAG_COL] == 1
        assert masked.equals(flagged), (
            "age missingness no longer corresponds one-to-one with "
            f"{AGE_FLAG_COL}; inspect before filling."
        )
        df.loc[masked, AGE_COL] = AGE_MASK_FILL
        print(
            f"Filled {int(masked.sum())} Safe Harbor-masked ages at "
            f"{AGE_MASK_FILL}; {AGE_FLAG_COL} retained to carry the censoring."
        )

    ############################################################################
    # Step 4b. Creatinine ratios
    #
    # KDIGO staging is built on the ratio of current creatinine to baseline.
    # Both inputs are fixed at randomization, and the transform is row-wise, so
    # there is no train/test contamination in constructing it here.
    #
    # Division guards against a zero denominator, which the deposit does not
    # currently contain but which would produce inf and survive silently into
    # the variance and missingness steps below.
    ############################################################################

    def _safe_ratio(numer, denom):
        out = numer / denom.replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    if {CREAT_AT_RAND, BASELINE_CREAT}.issubset(df.columns):
        df["creat_ratio_rand"] = _safe_ratio(df[CREAT_AT_RAND], df[BASELINE_CREAT])
        df["creat_delta_rand"] = df[CREAT_AT_RAND] - df[BASELINE_CREAT]

    if {CREAT_AT_RAND, MINCREAT_48}.issubset(df.columns):
        df["creat_ratio_min48"] = _safe_ratio(df[CREAT_AT_RAND], df[MINCREAT_48])

    print(
        "Derived: "
        + ", ".join(
            c
            for c in ("creat_ratio_rand", "creat_delta_rand", "creat_ratio_min48")
            if c in df.columns
        )
    )

    ############################################################################
    # Step 5. Zero Variance Columns
    ############################################################################
    # Select only numeric columns s/t .var() can be applied since you can only
    # call this function on numeric columns; otherwise, if you include a mix
    # (object and numeric), it will throw a FutureWarning about dropping
    # nuisance columns in DataFrame reductions.
    #
    # categorical_cols are excluded even when they hold integers, since their
    # variance is not meaningful and they are about to become category dtype.
    ############################################################################

    if stage == "training":
        numeric_cols = [
            c
            for c in df.select_dtypes(include=["number"]).columns
            if c not in categorical_cols
        ]
        var_indf = df[numeric_cols].var()

        # identify zero variance columns
        zero_var = var_indf[var_indf == 0]
        # capture zero-variance cols in list
        zero_varlist_list = list(zero_var.index)

        console.rule("[bold]Zero variance")
        print(f"Zero-variance columns: {zero_varlist_list}")

        ########################################################################
        # Step 5a. Save and Log Zero Variance Columns List
        ########################################################################

        dumpObjects(
            zero_varlist_list,
            os.path.join(data_path, "zero_varlist_list.pkl"),
        )

        if track:
            mlflow_dumpArtifact(
                experiment_name=exp_artifact_name,
                run_name=preproc_run_name,
                obj_name="zero_varlist_list",
                obj=zero_varlist_list,
            )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Zero Variance Columns List
        ########################################################################

        zero_varlist_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
        )

    ############################################################################
    # Step 6. Categorical dtype assignment
    #
    # Hoisted out of the training-only branch it sat in originally. Running it
    # in training alone produces a schema mismatch at inference: hospital comes
    # back as int64 rather than category, and any encoder fitted on the training
    # categories then sees a different type.
    ############################################################################

    for col in categorical_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) > 2:
                df[col] = (
                    df[col].astype(str).replace("nan", "Missing_NA").astype("category")
                )
        else:
            df[col] = df[col].fillna("Missing_NA").astype("category")

    ############################################################################
    # Step 7. Remove zero variance cols from main df, and assign to new var
    # df_sans_zero
    ############################################################################

    df_sans_zero = df.drop(columns=[c for c in zero_varlist_list if c in df.columns])

    print()
    print(f"Original shape: {df.shape[1]} columns.")
    print(f"Reduced by {df.shape[1] - df_sans_zero.shape[1]} zero variance columns.")
    print(f"Now there are {df_sans_zero.shape[1]} columns.")
    print()

    ############################################################################
    # Step 8. Handle Missing Data
    ############################################################################
    # 1. df_sans_zero.isnull().sum() counts the number of missing values in each
    #    column.
    # 2. len(df_sans_zero) gives the total number of rows in the DataFrame.
    # 3. Dividing gives the proportion of missing values.
    # 4. Multiplying by 100 converts this proportion into a percentage.
    ############################################################################

    console.rule("[bold]Missingness")

    if stage == "training":

        """
        Process Description: Handling Missing Data in a Dataset

        1. Identifying Missing Data by Column

        The first step in handling missing data involves calculating the
        percentage of missing values for each column in the dataset. This helps
        visualize the distribution of missing values and determine a reasonable
        threshold for column retention.
        """

        perc_missing_vals_per_col = (
            df_sans_zero.isnull().sum() / len(df_sans_zero)
        ) * 100

        """
        Filtering Columns Based on Missing Data Threshold

        Columns with more than miss_col_thresh percent missing values are
        removed; those below are retained.

        On the ELAIA-1 deposit nothing approaches the threshold. The worst
        offenders are the cost columns at 7.7%, and those are already gone as
        post-randomization leakage. The step is retained because the threshold
        is a contract with the inference stage, not because it is expected to
        bite here.
        """

        perc_below_indiv = perc_missing_vals_per_col[
            perc_missing_vals_per_col <= miss_col_thresh
        ].index.tolist()

        # Never let the missingness filter remove an outcome column.
        perc_below_indiv = sorted(
            set(perc_below_indiv) | (PROTECTED & set(df_sans_zero.columns)),
            key=list(df_sans_zero.columns).index,
        )

        print(
            f"Columns above {miss_col_thresh}% missing: "
            f"{sorted(set(df_sans_zero.columns) - set(perc_below_indiv))}"
        )
        print("\nTop 10 columns by percent missing:")
        print(perc_missing_vals_per_col.sort_values(ascending=False).head(10).round(2))

        dumpObjects(
            perc_below_indiv,
            os.path.join(data_path, "perc_below_indiv.pkl"),
        )

        if track:
            mlflow_dumpArtifact(
                experiment_name=exp_artifact_name,
                run_name=preproc_run_name,
                obj_name="perc_below_indiv",
                obj=perc_below_indiv,
            )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Percentage Below Threshold List
        ########################################################################

        perc_below_indiv = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="perc_below_indiv",
        )

    ############################################################################
    # Step 9. Apply the retention list
    #
    # The original template computed perc_below_indiv, pickled it, and then
    # never applied it, so the threshold had no effect on the output. Applying
    # it here also pins column order, which is what makes the training and
    # inference frames line up.
    ############################################################################

    absent = [c for c in perc_below_indiv if c not in df_sans_zero.columns]
    if absent:
        raise ValueError(
            f"Columns retained at training are absent at {stage}: {absent}"
        )

    unexpected = [c for c in df_sans_zero.columns if c not in perc_below_indiv]
    if unexpected and stage == "inference":
        print(f"Dropping columns not seen at training: {unexpected}")

    df_sans_zero = df_sans_zero[perc_below_indiv]

    ############################################################################
    # Step 10. Row-level missingness feature
    #
    # Computed on the retained columns so the denominator matches between
    # training and inference. Order matters: computing it before Step 9 would
    # make the value depend on columns that are about to disappear.
    ############################################################################

    df_sans_zero[percent_miss] = df_sans_zero.isna().mean(axis=1)

    ############################################################################
    # Step 11. Save Processed Data
    ############################################################################

    console.rule("[bold]Output")
    print(f"Final shape: {df_sans_zero.shape}")
    print(f"Outcome columns present: {sorted(PROTECTED & set(df_sans_zero.columns))}")
    print(f"Writing to {output_data_file}")

    os.makedirs(os.path.dirname(output_data_file) or ".", exist_ok=True)
    df_sans_zero.to_parquet(output_data_file)


if __name__ == "__main__":
    app()
