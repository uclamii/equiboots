import os
import numpy as np

from pathlib import Path

import mlflow
from dotenv import load_dotenv
from loguru import logger

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import sys

sys.path.insert(0, "fistula_graft_implants")

from constants import (
    exp_artifact_name,
    preproc_run_name,
)
from functions import mlflow_loadArtifact

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = Path(__file__).resolve().parents[1]  # fistula_graft_implants/  (for imports)
PROJECT_DIR = Path(__file__).resolve().parents[2]  # Fistula_Graft_implants/
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR_INFER = DATA_DIR / "processed/inference"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
RESULTS_DIR = MODELS_DIR / "results"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

features_path = PROCESSED_DATA_DIR / "X.parquet"

################################################################################
############################ MLflow Tracking Backend ###########################
################################################################################
# MLflow 3.x refuses the bare filesystem store ('./mlruns') and raises rather
# than falling back, which is what produced the import-time failure. The URI is
# resolved once here so every reader and writer in the project agrees on it.
#
# Anchored to PROJ_ROOT rather than the working directory. A relative
# 'sqlite:///mlflow.db' resolves against wherever the process happened to start,
# so a target invoked from a subdirectory would silently create a second, empty
# database instead of finding the real one.
#
# Override in .env or the environment:
#   MLFLOW_TRACKING_URI=http://mlflow.internal:5000
#
# To migrate existing runs out of mlruns/ (lossless, keeps run history):
#   mlflow migrate-filestore --source . --target sqlite:///mlflow.db
# Do not delete mlruns/ afterwards; only the metadata moves, and the logged
# artifacts stay on disk at their original paths.
################################################################################

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJ_ROOT / 'mlflow.db'}",
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

################################################################################
############################ Global Constants ##################################
################################################################################

rstate = 222  # random state for reproducibility
threshold_target_metric = "precision"  # target metric for threshold optimization
target_precision = 0.8  # target precision for threshold optimization

sampler_definitions = {
    "None": None,
    "RandomUnderSampler": RandomUnderSampler(random_state=rstate),
    "RandomOverSampler": RandomOverSampler(random_state=rstate),
}

rfe_estimator = LogisticRegression(
    max_iter=100,
    n_jobs=-2,
)

# Remove 10% of features per iteration
rfe = RFE(
    estimator=rfe_estimator,
    step=0.1,
)


################################################################################
# This section here is for categorical variables

categorical_cols = [col.lower() for col in []]

################################################################################
######################### Feature Column Resolution ############################
################################################################################
# The original block loaded X_columns_list from MLflow at import, caught every
# exception, and fell back to []. Three things went wrong with that.
#
# 1. config.py is imported by preprocessing.py, which runs BEFORE feat_gen.py
#    has logged X_columns_list. On a fresh clone the artifact cannot exist yet,
#    so the lookup fails every time by construction, not by accident.
#
# 2. The failure was logged at ERROR and then swallowed. X_columns_list became
#    [], numerical_cols became [], and the ColumnTransformer was built with an
#    empty column list for both transformers. That is a preprocessor that does
#    nothing, and it fits and transforms without complaint.
#
# 3. Nothing downstream could tell the difference between "no features yet" and
#    "no features, ever".
#
# The fix keeps MLflow as the source of truth and adds a disk fallback: X.parquet
# is what feat_gen.py just wrote, and its schema can be read without loading the
# frame. Callers that genuinely need the columns call require_feature_columns()
# and get an exception instead of a silent no-op.
################################################################################


def _feature_columns_from_mlflow():
    """Return the logged feature list, or None if it is not available."""
    try:
        cols = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,  # Use the same run_name as training
            obj_name="X_columns_list",
            verbose=False,
        )
    except Exception as e:
        logger.debug(f"X_columns_list not available from MLflow: {e}")
        return None

    if not cols:
        logger.debug("X_columns_list from MLflow was empty or None.")
        return None

    return list(cols)


def _feature_columns_from_parquet(path=features_path):
    """
    Return the column names of X.parquet without reading the data.

    pyarrow reads the footer only, so this stays cheap even on a large feature
    matrix. Falls back to a zero-row pandas read if pyarrow is unavailable.
    """
    if not Path(path).exists():
        return None

    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except ImportError:
        import pandas as pd

        return list(pd.read_parquet(path).columns)
    except Exception as e:
        logger.debug(f"Could not read schema from {path}: {e}")
        return None


def resolve_feature_columns():
    """
    Resolve the feature column list, preferring the logged contract.

    MLflow first, because at inference the logged list is the contract that
    keeps the frame aligned with the fitted model. X.parquet second, because
    during a training run it is the freshest thing on disk and MLflow has
    nothing yet.
    """
    cols = _feature_columns_from_mlflow()
    if cols:
        logger.info(f"Feature columns from MLflow: {len(cols)}")
        return cols

    cols = _feature_columns_from_parquet()
    if cols:
        logger.info(f"Feature columns from {features_path}: {len(cols)}")
        return cols

    logger.warning(
        "No feature columns available from MLflow or "
        f"{features_path}. This is expected before feat_gen.py has run; "
        "anything that builds a preprocessor now will get an empty column "
        "list. Call require_feature_columns() to fail loudly instead."
    )
    return []


def require_feature_columns():
    """
    Same as resolve_feature_columns(), but raises when nothing is found.

    Use this from training and inference entry points. A preprocessor fitted on
    an empty column list is worse than a crash, because it succeeds.
    """
    cols = resolve_feature_columns()
    if not cols:
        raise RuntimeError(
            "No feature columns found. Checked MLflow "
            f"(experiment='{exp_artifact_name}', run='{preproc_run_name}', "
            f"tracking_uri='{MLFLOW_TRACKING_URI}') and {features_path}. "
            "Run the preprocessing and feat_gen training stages first."
        )
    return cols


X_columns_list = resolve_feature_columns()

# Subset the numerical columns only; categorical columns are already defined above
numerical_cols = [col for col in X_columns_list if col not in categorical_cols]


################################################################################
############################### Transformers ###################################
################################################################################
# Impute, then scale. The original order scaled first, which happens to survive
# because StandardScaler ignores NaN and passes it through, but it makes the
# imputed value depend on the scaler's fitted statistics rather than the raw
# column. It also breaks outright the moment the scaler is swapped for one that
# does not tolerate NaN, and it makes strategy='median' mean something other
# than the column median.
################################################################################

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


################################################################################
# build_preprocessor() exists so callers that resolve their columns late (an
# inference script that just loaded the logged contract, a notebook, a test) can
# construct a correct ColumnTransformer instead of inheriting whatever the
# module-level one captured at import time.
################################################################################


def build_preprocessor(numerical_cols=None, categorical_cols=categorical_cols):
    """Create the ColumnTransformer with passthrough."""
    if numerical_cols is None:
        numerical_cols = globals()["numerical_cols"]

    if not numerical_cols and not categorical_cols:
        logger.warning(
            "build_preprocessor() called with no columns; the resulting "
            "ColumnTransformer will be a no-op."
        )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        # remainder="passthrough",
        # prevents prepending transformer names (e.g., 'remainder_') to output
        # feature names
        # verbose_feature_names_out=False,
    )


preprocessor = build_preprocessor(numerical_cols, categorical_cols)

################################################################################
################################ Pipelines #####################################
################################################################################

pipeline_scale_imp_rfe = [
    ("Preprocessor", preprocessor),
    ("RFE", rfe),
]

pipeline_scale_imp = [
    ("Preprocessor", preprocessor),
]

pipelines = {
    "orig": {
        "pipeline": pipeline_scale_imp,
        "sampler": None,
        "feature_selection": False,  # No feature selection for orig
    },
    "under": {
        "pipeline": pipeline_scale_imp,
        "sampler": RandomUnderSampler(random_state=rstate),
        "feature_selection": False,  # No feature selection for under
    },
    "over": {
        "pipeline": pipeline_scale_imp,
        "sampler": RandomOverSampler(random_state=rstate),
        "feature_selection": False,  # No feature selection for under
    },
    "orig_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": None,
        "feature_selection": True,  # Feature selection (RFE) for orig_rfe
    },
    "under_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": RandomUnderSampler(random_state=rstate),
        "feature_selection": True,  # Feature selection (RFE) for under_rfe
    },
    "over_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": RandomOverSampler(random_state=rstate),
        "feature_selection": True,  # Feature selection (RFE) for over_rfe
    },
}


################################################################################
# RFE hyperparameters are keyed 'feature_selection_RFE__*' because model_tuner
# renames any sklearn.feature_selection step to f"feature_selection_{name}"
# before handing the grid to the search. That naming is correct as written.
#
# What is not safe is reusing the same grid for the non-RFE pipelines above:
# 'orig', 'under', and 'over' have no RFE step, so an
# 'feature_selection_RFE__n_features_to_select' key has nothing to bind to.
# strip_rfe_params() removes those keys for a pipeline that does not select.
################################################################################


def strip_rfe_params(tuned_parameters):
    """Drop feature_selection_* keys from a parameter grid."""
    return [
        {k: v for k, v in grid.items() if not k.startswith("feature_selection_")}
        for grid in tuned_parameters
    ]


def params_for(tuned_parameters, feature_selection):
    """Return the grid appropriate to a pipeline with or without RFE."""
    return tuned_parameters if feature_selection else strip_rfe_params(tuned_parameters)


################################################################################
############################# Path Variables ###################################
################################################################################

# model_output = "model_output"  # model output path
# mlflow_data = "mlflow_data"  # path to store mlflow artificats (i.e., results)

################################################################################
########################## Logistic Regression #################################
################################################################################

# Define the hyperparameters for Logistic Regression
lr_name = "lr"

# lr_penalties = ["elasticnet"]
# lr_penalties = ["l1"]
lr_penalties = ["l2"]
lr_Cs = np.logspace(-4, 0, 10)
l1_ratio = np.linspace(0, 1, 10)

# Structure the parameters similarly to the RF template
tuned_parameters_lr = [
    {
        "lr__penalty": lr_penalties,
        "lr__C": lr_Cs,
        "feature_selection_RFE__n_features_to_select": [10, 0.1, 0.5, 0.7, 1.0],
        # "lr__l1_ratio": l1_ratio,
    }
]

lr = LogisticRegression(
    class_weight="balanced",
    random_state=rstate,
    n_jobs=-2,
    # solver="saga",
    # solver="liblinear",
    solver="lbfgs",
)

lr_definition = {
    "clc": lr,
    "estimator_name": lr_name,
    "tuned_parameters": tuned_parameters_lr,
    "randomized_grid": True,
    "n_iter": 10,
    "early": False,
}


################################################################################
########################## Random Forest Classifier ############################
################################################################################

# Define the hyperparameters for Random Forest
rf_name = "rf"

rf_n_estimators = [100, 200, 300]
rf_max_depths = [None, 5, 10]
rf_criterions = ["gini", "entropy"]
rf_parameters = [
    {
        "rf__n_estimators": rf_n_estimators,
        "rf__max_depth": rf_max_depths,
        "rf__criterion": rf_criterions,
        "feature_selection_RFE__n_features_to_select": [10, 0.1, 0.5, 0.7, 1.0],
    }
]

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=rstate,
    n_jobs=-2,
)

rf_definition = {
    "clc": rf,
    "estimator_name": rf_name,
    "tuned_parameters": rf_parameters,
    "randomized_grid": True,
    "n_iter": 10,
    "early": False,
}

################################################################################
############################## XGBoost Classifier ##############################
################################################################################

# Estimator name prefix for use in GridSearchCV or similar tools
xgb_name = "xgb"

################################################################################
# device="cuda" raises on a CPU-only host, and the fistula pipeline gets run on
# both. Driven from the environment so the same config works in either place:
#   XGB_DEVICE=cpu make train_pipeline
# Default is unchanged.
################################################################################

XGB_DEVICE = os.getenv("XGB_DEVICE", "cuda")

xgb = XGBClassifier(
    objective="binary:logistic",
    random_state=rstate,
    tree_method="hist",
    device=XGB_DEVICE,  # cuda for gpu
    n_jobs=16,
    enable_categorical=True,
)

# Define the hyperparameters for XGBoost
xgb_learning_rates = [0.001]  # Learning rate or eta
xgb_n_estimators = [10000]  # Number of trees. Equivalent to n_estimators in GB
xgb_max_depths = [3, 5, 7]  # Maximum depth of the trees
xgb_subsamples = [0.8, 1.0]  # Subsample ratio of the training instances
xgb_colsample_bytree = [0.8, 1.0]
xgb_alpha = [0, 0.1, 1, 10]  # L1 regularization (alpha)
xgb_lambda = [0, 0.1, 10, 100]  # L2 regularization (lambda)
xgb_eval_metric = ["logloss"]  # check out "aucpr"
xgb_early_stopping_rounds = [3]
xgb_verbose = [0]
# Subsample ratio of columns when constructing each tree

# Combining the hyperparameters in a dictionary
xgb_parameters = [
    {
        "xgb__learning_rate": xgb_learning_rates,
        "xgb__n_estimators": xgb_n_estimators,
        "xgb__max_depth": xgb_max_depths,
        "xgb__subsample": xgb_subsamples,
        "xgb__alpha": xgb_alpha,  # L1 regularization (alpha)
        "xgb__lambda": xgb_lambda,  # L2 regularization (lambda)
        "xgb__colsample_bytree": xgb_colsample_bytree,
        "xgb__eval_metric": xgb_eval_metric,
        "xgb__early_stopping_rounds": xgb_early_stopping_rounds,
        "xgb__verbose": xgb_verbose,
        "feature_selection_RFE__n_features_to_select": [10, 0.1, 0.5, 0.7, 1.0],
    }
]

xgb_definition = {
    "clc": xgb,
    "estimator_name": xgb_name,
    "tuned_parameters": xgb_parameters,
    "randomized_grid": True,
    "n_iter": 10,
    "early": True,
}

################################################################################
############################ CatBoost Classifier ###############################
################################################################################

cat_name = "cat"

cat = CatBoostClassifier(
    task_type="CPU",
    random_state=rstate,
    eval_metric="Logloss",
    thread_count=8,
)

# Define the hyperparameters for CatBoost
cat_depths = [4, 6, 8, 10]  # Depth of the trees
cat_learning_rates = [0.001]  # Learning rate
cat_l2_leaf_regs = [3, 10, 100]  # L2 regularization
cat_bagging_temperatures = [0, 0.5, 1]  # Bagging temperature
cat_n_estimators = [10000]  # Number of trees
cat_early_stopping_rounds = [3]  # Early stopping rounds
cat_random_strengths = [1, 10]  # Random strength for feature score randomness
cat_verbose = [0]  # Verbosity level
cat_n_features_to_select = [10, 0.1, 0.5, 0.7, 1.0]  # Features to select for RFE

# Combining the hyperparameters in a dictionary
cat_parameters = [
    {
        "cat__depth": cat_depths,
        "cat__learning_rate": cat_learning_rates,
        "cat__l2_leaf_reg": cat_l2_leaf_regs,
        "cat__bagging_temperature": cat_bagging_temperatures,
        "cat__n_estimators": cat_n_estimators,
        "cat__early_stopping_rounds": cat_early_stopping_rounds,
        "cat__random_strength": cat_random_strengths,
        "cat__verbose": cat_verbose,
        "feature_selection_RFE__n_features_to_select": cat_n_features_to_select,
    }
]

cat_definition = {
    "clc": cat,
    "estimator_name": cat_name,
    "tuned_parameters": cat_parameters,
    "randomized_grid": True,
    "n_iter": 10,
    "early": True,
}


model_definitions = {
    lr_name: lr_definition,
    rf_name: rf_definition,
    xgb_name: xgb_definition,
    cat_name: cat_definition,
}
