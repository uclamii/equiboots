from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scipy.stats as stats

import xgboost

print(f"XGBoost version: {xgboost.__version__}")

# fetch dataset
adult = fetch_ucirepo(id=2)
adult = adult.data.features.join(adult.data.targets, how="inner")

adult.head(3)

# Drop missing values
adult.dropna(inplace=True)

df = adult.copy()

df["race"] = df["race"].replace("Amer-Indian-Eskimo", "Native American or Inuit")


def outcome_merge(val):
    if val == "<=50K" or val == "<=50K.":
        return 0
    else:
        return 1


df["income"] = df["income"].apply(outcome_merge)

#  sex, count and percentages above_50k

income_by_sex = df.groupby("sex")["income"].agg(
    ["count", lambda x: (x.sum() / x.count()) * 100]
)
income_by_sex.columns = ["count", "percentage_above_50k"]
print(income_by_sex)

## Split the data ##############################################################

X = df.drop("income", axis=1)
y = df["income"]

for col in X.columns:
    if isinstance(X[col], object):
        X[col] = X[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

## Model #######################################################################

model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    enable_categorical=True,
)
model.fit(X_train, y_train)

print(model)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print(y_pred[:10])
print(y_prob[:10])

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
