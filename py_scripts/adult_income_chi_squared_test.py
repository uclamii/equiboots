from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

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
    if X[col].dtype == "object":
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

################################################################################
### EquiBoots Tests
################################################################################

import equiboots as eqb
from equiboots import EquiBoots

# EquiBoots expects index-aligned arrays
y_true = y_test.reset_index(drop=True).values
fairness_df = X_test[["race"]].reset_index(drop=True)
fairness_df["race"] = fairness_df["race"].astype(str)

# Binary classification: pass the positive-class probability as 1D
y_prob_pos = y_prob[:, 1]

eq = EquiBoots(
    y_true=y_true,
    y_prob=y_prob_pos,
    y_pred=y_pred,
    fairness_df=fairness_df,
    fairness_vars=["race"],
    reference_groups=["White"],
    task="binary_classification",
    bootstrap_flag=False,
    balanced=True,
    stratify_by_outcome=False,
    group_min_size=50,
)

eq.grouper(groupings_vars=["race"])
data = eq.slicer("race")

print("Below min size:", eq.groups_below_min_size)

# Per-group metrics dict (includes TP, FP, TN, FN plus rate metrics)
race_metrics = eq.get_metrics(data)

# Readable table view
race_metrics_df = eqb.metrics_dataframe(metrics_data=race_metrics)
print("\nPer-group metrics:")
print(race_metrics_df)

# Now run chi-square against the White reference group
test_config = {
    "test_type": "chi_square",
    "alpha": 0.05,
    "adjust_method": "bonferroni",
    "confidence_level": 0.95,
    "classification_task": "binary_classification",
}

stat_test_results = eq.analyze_statistical_significance(
    race_metrics, "race", test_config
)

# Raw per-group metrics table (CM cells + computed rates)
rows = []
for group, m in race_metrics.items():
    tp, fp, tn, fn = m.get("TP", 0), m.get("FP", 0), m.get("TN", 0), m.get("FN", 0)
    rows.append(
        {
            "Group": group,
            "n": tp + fp + tn + fn,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Accuracy": m.get("Accuracy"),
            "Precision": m.get("Precision"),
            "Recall": m.get("Recall"),
            "Specificity": m.get("Specificity"),
            "F1 Score": m.get("F1 Score"),
            "FP Rate": m.get("FP Rate"),
            "FN Rate": m.get("FN Rate"),
            "Predicted Prevalence": m.get("Predicted Prevalence"),
        }
    )

raw_df = pd.DataFrame(rows).set_index("Group")
print("\nRaw per-group metrics:")
print(raw_df.to_string(float_format=lambda x: f"{x:.4f}"))

print("\nChi-square results:")
for outer_key, inner in stat_test_results.items():
    print(f"\n=== {outer_key} ===")
    for metric, result in inner.items():
        sig = "*" if result.is_significant else " "
        es = (
            f", Cramer's V={result.effect_size:.3f}"
            if result.effect_size is not None
            else ""
        )
        print(
            f"  {sig} {metric:25s} chi2={result.statistic:8.3f}  p={result.p_value:.4g}{es}"
        )
