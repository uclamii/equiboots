"""
Reproduction: equiboots.metrics_table ignores `reference_group` and drops round().

    pip install equiboots==0.0.1a14
    python repro_metrics_table.py

Data: UCI Adult / census income, fetched at runtime. Nothing to download by hand.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from equiboots import EquiBoots, metrics_table

COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
URL = "https://raw.githubusercontent.com/shap/shap/master/data/adult.data"
df = pd.read_csv(URL, names=COLS, skipinitialspace=True, na_values="?").dropna()

y = (df["income"] == ">50K").astype(int)
X = pd.get_dummies(df.drop(columns=["income"]), drop_first=True)

X_tr, X_te, y_tr, y_te, sex_tr, sex_te = train_test_split(
    X, y, df["sex"], test_size=0.3, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
    X_tr, y_tr
)
y_prob = model.predict_proba(X_te)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
y_true = y_te.to_numpy()

# Group keys are "Male" and "Female". reset_index is required: equiboots uses
# pandas index labels to index numpy arrays positionally.
fairness_df = sex_te.to_frame().reset_index(drop=True)

eq = EquiBoots(
    y_true=y_true,
    y_prob=y_prob,
    y_pred=y_pred,
    fairness_df=fairness_df,
    fairness_vars=["sex"],
)
eq.grouper(groupings_vars=["sex"])
sex_metrics = eq.get_metrics(eq.slicer("sex"))

print("group keys:", list(sex_metrics))
print()

# BUG 1: "male" is a case typo for the "Male" key.
# Before the fix this returns a table. After the fix it raises.
print("--- invalid reference_group ---")
try:
    metrics_table(sex_metrics, reference_group="male")
    print("FAIL (bug present): no error raised for reference_group='male'")
except ValueError as e:
    print(f"PASS (fixed): {e}")
print()

# BUG 2: decimal_places=3 requested.
# Before the fix values come back at full precision, since round() is discarded.
print("--- valid reference_group ---")
table = metrics_table(sex_metrics, reference_group="Male", decimal_places=3)
print(table.loc[["Accuracy", "Precision", "Recall"]])

shown = str(table.loc["Accuracy", "Male"])
status = "PASS (fixed)" if len(shown.split(".")[-1]) <= 3 else "FAIL (bug present)"
print()
print(f"{status} - Accuracy['Male'] = {shown}")
