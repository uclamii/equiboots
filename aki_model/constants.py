"""
Constants for the ELAIA-1 AKI alert preprocessing stage.

Source: Wilson FP et al. Electronic health record alerts for acute kidney
injury: multicenter, randomized clinical trial. BMJ 2021;372:m4786.
Deposit: Dryad 10.5061/dryad.4f4qrfj95 (6,030 randomized adult inpatients).

Every drop rule in this file carries its reason as the dict value. preprocess.py
applies them through should_drop() and reports the tally by reason, so the drop
list stays auditable without anyone having to re-derive the logic.
"""

################################################################################
########################## Variable/DataFrame Constants ########################
################################################################################

import os

var_index = "id"

################################################################################

# The below artificat name is used for preprocessing alone
exp_artifact_name = "preprocessing"
preproc_run_name = "preprocessing"
artifact_run_id = "preprocessing"
artifact_name = "preprocessing"

miss_col_thresh = 60  # missingness threshold tolerated for zero-var cols
perc_below_indiv = f"perc_below_{miss_col_thresh}_indiv"
percent_miss = "percentage_missing"  # new col for percentage missing in rows
miss_indicator = "missing_indicator"  # indicator for percentage missing (0,1)

################################################################################
############################# Mlflow Variables #################################
################################################################################

mlflow_artifacts_data = "./mlruns/preprocessing"
mlflow_models_data = "./mlruns/models"
mlflow_models_copy = "./mlruns/models_copy"

artifact_data = "artifacts/"  # path to store mlflow artifacts
profile_data = "profile_data"  # path to store pandas profiles in
data_path = "data/processed/"

## DataBricks
databricks_username = "/" + "/".join(os.getcwd().split("/")[2:-1]) + "/"

exp_artifact_name

################################################################################
########################## The randomization boundary ##########################
################################################################################
# ELAIA-1 is a randomized trial. Every drop rule below exists to enforce one
# line: a feature is admissible only if its value was fixed at or before the
# moment of randomization. Anything measured afterwards is post-treatment. It
# is downstream of the intervention under study and of the outcome itself, so
# including it produces a model that reads the future rather than predicts it.
################################################################################

################################################################################
# PROTECTED
#
# The outcome family. These are never dropped here. X and y are established in
# feat_gen.py, which selects the label and removes the rest. Carrying all of
# them through preprocessing keeps both framings open downstream: the binary
# composite, any single component, or a time-to-event formulation.
#
# Verified on all 6,030 rows:
#   composite_outcome == (aki_progression14 | death14 | dialysis14)
################################################################################

PROTECTED = {
    "composite_outcome",
    "aki_progression14",
    "death14",
    "dialysis14",
    "time_to_composite_outcome",
    "time_to_aki_progression",
    "time_to_death",
    "time_to_dialysis",
}

outcomes = (
    "composite_outcome",
    "aki_progression14",
    "death14",
    "dialysis14",
    "time_to_composite_outcome",
    "time_to_aki_progression",
    "time_to_death",
    "time_to_dialysis",
)

target = "composite_outcome"

################################################################################
# LEAKAGE
#
# Post-randomization columns caught by name rather than by pattern.
################################################################################

LEAKAGE = {
    # ---- care processes in the same 14-day window as the outcome ----------
    "consult14": "leakage: 14-day care process, same window as outcome",
    "time_to_consult": "leakage: 14-day care process, same window as outcome",
    # ---- quantities that resolve only at or after discharge ---------------
    "max_stage": (
        "leakage: max AKI stage over the encounter; AKI progression is defined "
        "as a KDIGO stage above the stage at randomization, so this is close to "
        "the label itself"
    ),
    "aki_duration": "leakage: spans post-randomization, resolves at recovery",
    "los_since_alert": "leakage: length of stay measured from randomization",
    "discharge_to_home": "leakage: disposition, known only at discharge",
    "direct_cost": "leakage: billing, accrues over the whole encounter",
    "total_cost": "leakage: billing, accrues over the whole encounter",
    "aki_documentation": (
        "leakage: discharge ICD coding of this AKI episode, assigned after the "
        "outcome window"
    ),
    # ---- alert-process variables -----------------------------------------
    # These are populated in both arms, which makes them look like baseline
    # covariates. They are not. Each accumulates over the alert episode and is
    # therefore fixed only after randomization.
    "attending": "leakage: alert-process, accrues over the episode",
    "unique_providers": "leakage: alert-process, accrues over the episode",
    "duration": "leakage: alert-process, categorized alert duration",
    "duration_of_alert": "leakage: alert-process, accrues over the episode",
    "other_alert_burden": "leakage: alert-process, accrues over the episode",
}

################################################################################
# PREFIXES
################################################################################

PREFIXES = {
    "delta_": "leakage: post-randomization change score",
}

################################################################################
# SUFFIXES
#
# The naming convention in this deposit is load-bearing. Everything ending in
# post24 / post28 is a medication, procedure, or measurement recorded in the
# hours after randomization; the _post_NN_ vitals are the same. Note that the
# pre24 and prior_ families are the mirror image and are retained.
################################################################################

SUFFIXES = {
    "post24": "leakage: exposure within 24h after randomization",
    "post28": "leakage: measurement within 28h after randomization",
    "_post_24_min": "leakage: vital sign minimum, 24h after randomization",
    "_post_24_max": "leakage: vital sign maximum, 24h after randomization",
    "_post_48_min": "leakage: vital sign minimum, 48h after randomization",
}

################################################################################
# REFERENCE_LEVELS
#
# Placeholder, kept for parity with the pipeline this template came from. The
# ELAIA-1 deposit arrives with no one-hot expansions, so there are no reference
# levels to collapse yet. Populate this when hospital is expanded downstream.
################################################################################

REFERENCE_LEVELS = {}

################################################################################
######################### De-identification artifacts ##########################
################################################################################
# age is blank for exactly the 507 patients flagged age_over_90 == 1, and for
# no one else. This is the HIPAA Safe Harbor mask on ages 90 and above, not
# data loss. Median imputation would pull the oldest and highest-risk patients
# toward the centre of the distribution, so the mask is filled at the floor of
# the censored band and age_over_90 is retained to carry the censoring.
################################################################################

AGE_COL = "age"
AGE_FLAG_COL = "age_over_90"
AGE_MASK_FILL = 90.0

################################################################################
######################## Derived creatinine ratio inputs #######################
################################################################################
# KDIGO staging is built on the ratio of current creatinine to baseline. Tree
# models recover it; linear models do not. Constructed here because it is a
# deterministic row-wise transform of columns already fixed at randomization,
# which means no train/test contamination.
################################################################################

CREAT_AT_RAND = "creat_at_rand"
BASELINE_CREAT = "baseline_creat"
MINCREAT_48 = "mincreat48"

# Multi-level nominal columns.
categorical_cols = ["hospital"]

# Attributes available for subgroup performance and fairness auditing.
sensitive_cols = ["race", "ethnicity", "sex", "age"]

# Treatment assignment. A design variable, not a risk factor.
# feat_gen.py decides whether it enters X.
treatment_col = "alert"
