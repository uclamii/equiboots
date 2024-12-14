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
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from model_tuner import Model

############################### Load The Dataset ###############################

### fetch dataset
adult = fetch_ucirepo(id=2)

##################### Define the Feature Space and Outcome #####################
X = adult.data.features
y = adult.data.targets

print("-" * 80)
print("X")
print("-" * 80)

print(X.head())  # inspect first 5 rows of X

print("-" * 80)
print("y = Outcome = Income")
print("-" * 80)

print(f"\n{y.head()}")  # inspect first 5 rows of y

y.loc[:, "income"] = y["income"].str.rstrip(".")  # Remove trailing periods

print(f"\n Income Value Counts: \n")

# Check the updated value counts
print(y["income"].value_counts())

y = y["income"].map({"<=50K": 0, ">50K": 1})

outcome = ["y"]

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

y_prob = pd.DataFrame(y_prob)

print(f"Predicted Probabilities: \n {y_prob}")


y_pred = model_xgb.predict(X_test, optimal_threshold=True)

# Cast predictions into DataFrame
y_pred = pd.DataFrame(y_pred)

print(f"Predictions: \n {y_pred}")
