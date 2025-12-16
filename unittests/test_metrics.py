import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.equiboots import metrics, root_mean_squared_error

from src.equiboots.metrics import (
    binary_classification_metrics,
    multi_class_prevalence,
    multi_class_classification_metrics,
    multi_label_classification_metrics,
    regression_metrics,
    metrics_dataframe,
    mean_squared_error,
    calculate_bootstrap_stats,
)


def test_binary_classification_example_executes():
    metrics.binary_classification_example()


def test_multi_class_classification_example_executes():
    metrics.multi_class_classification_example()


def test_multi_label_classification_example_executes():
    metrics.multi_label_classification_example()


def test_regression_example_executes():
    metrics.regression_example()


def test_binary_classification_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.7, 0.8, 0.3, 0.9])
    y_pred = (y_proba > 0.5).astype(int)

    result = binary_classification_metrics(y_true, y_pred, y_proba)
    assert isinstance(result, dict)
    assert "Accuracy" in result
    assert result["TP Rate"] >= 0 and result["FP Rate"] >= 0


def test_multi_class_prevalence():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 0])
    prevalence, predicted_prevalence = multi_class_prevalence(
        y_true,
        y_pred,
        3,
    )
    assert len(prevalence) == 3
    assert len(predicted_prevalence) == 3
    assert np.isclose(sum(prevalence), 1.0)
    assert np.isclose(sum(predicted_prevalence), 1.0)


def test_multi_class_classification_metrics():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 0])
    y_proba = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.2, 0.7],
            [0.05, 0.1, 0.85],
            [0.2, 0.7, 0.1],
            [0.85, 0.1, 0.05],
        ]
    )
    result = multi_class_classification_metrics(
        y_true,
        y_pred,
        y_proba,
        n_classes=3,
    )
    assert "Accuracy" in result
    assert "ROC AUC" in result
    assert "Average Precision Score" in result


def test_multi_label_classification_metrics():
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform([[0], [1], [2], [0, 2], [1, 2]])
    y_pred = mlb.transform([[0], [1], [2], [2], [1]])
    y_proba = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.2, 0.8, 0.3],
            [0.1, 0.4, 0.9],
            [0.5, 0.3, 0.8],
            [0.2, 0.7, 0.6],
        ]
    )
    result = multi_label_classification_metrics(y_true, y_pred, y_proba)
    assert "Accuracy" in result
    assert "ROC AUC" in result
    assert isinstance(result["Prevalence multi-labels"], list)


def test_regression_metrics():
    y_true = np.array([3.0, 5.0, 2.0, 7.0])
    y_pred = np.array([2.5, 5.0, 4.0, 8.0])
    result = regression_metrics(y_true, y_pred)

    assert "Mean Absolute Error" in result
    assert "Root Mean Squared Error" in result
    assert "Mean Squared Log Error" in result
    assert result["R^2 Score"] <= 1.0


def test_main_executes_examples(monkeypatch):
    # Monkeypatch print to avoid clutter
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    import runpy

    runpy.run_module("src.equiboots.metrics", run_name="__main__")


def test_metrics_dataframe_outputs_correct_format():
    # Sample input
    input_data = [
        {
            "GroupA": {"Accuracy": 0.9, "F1 Score": 0.85},
            "GroupB": {"Accuracy": 0.8, "F1 Score": 0.75},
        },
        {
            "GroupA": {"Accuracy": 0.92, "F1 Score": 0.88},
            "GroupB": {"Accuracy": 0.78, "F1 Score": 0.70},
        },
    ]

    df = metrics_dataframe(input_data)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"Accuracy", "F1 Score", "attribute_value"}
    assert df.shape[0] == 4  # 2 groups * 2 timepoints = 4 rows
    assert all(df["attribute_value"].isin(["GroupA", "GroupB"]))
    assert df["Accuracy"].between(0, 1).all()


def test_root_mean_squared_error_equivalence():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    expected_rmse = mean_squared_error(y_true, y_pred, squared=False)
    actual_rmse = root_mean_squared_error(y_true, y_pred)

    assert np.isclose(
        expected_rmse, actual_rmse
    ), "RMSE fallback does not match expected output"


def test_calculate_bootstrap_stats_basic():
    rng = np.random.RandomState(0)
    n = 100

    # Build 100 bootstrap samples, each with A and B
    samples = []
    a_vals = rng.beta(2, 5, size=n)  # deterministic given seed
    b_vals = rng.beta(5, 2, size=n)
    for i in range(n):
        samples.append(
            {
                "A": {"Accuracy": float(a_vals[i])},
                "B": {"Accuracy": float(b_vals[i])},
            }
        )

    df = calculate_bootstrap_stats(samples, metric="Accuracy")

    # Expect rows for both groups
    assert set(df["group"]) == {"A", "B"}
    # Compute expected stats with numpy for robustness
    exp = {
        "A": {
            "mean": float(np.mean(a_vals)),
            "std": float(np.std(a_vals)),
            "lo": float(np.percentile(a_vals, 2.5)),
            "hi": float(np.percentile(a_vals, 97.5)),
            "n": len(a_vals),
        },
        "B": {
            "mean": float(np.mean(b_vals)),
            "std": float(np.std(b_vals)),
            "lo": float(np.percentile(b_vals, 2.5)),
            "hi": float(np.percentile(b_vals, 97.5)),
            "n": len(b_vals),
        },
    }
    for _, row in df.iterrows():
        g = row["group"]
        assert row["n_samples"] == exp[g]["n"]
        assert row["mean"] == pytest.approx(exp[g]["mean"], rel=1e-12)
        assert row["std"] == pytest.approx(exp[g]["std"], rel=1e-12)
        assert row["ci_lower"] == pytest.approx(exp[g]["lo"], rel=1e-12)
        assert row["ci_upper"] == pytest.approx(exp[g]["hi"], rel=1e-12)


def test_calculate_bootstrap_stats_handles_missing_metric_entries():
    # 5 samples where group B is missing the metric in two of them
    samples = [
        {"A": {"Accuracy": 0.8}, "B": {"Accuracy": 0.7}},
        {"A": {"Accuracy": 0.9}},  # B missing
        {"A": {"Accuracy": 0.85}, "B": {"Accuracy": 0.65}},
        {"A": {"Accuracy": 0.80}},  # B missing
        {"A": {"Accuracy": 0.95}, "B": {"Accuracy": 0.75}},
    ]
    df = calculate_bootstrap_stats(samples, metric="Accuracy").set_index("group")

    # A should have 5 samples, B should have 3
    assert df.loc["A", "n_samples"] == 5
    assert df.loc["B", "n_samples"] == 3

    a_vals = np.array([0.8, 0.9, 0.85, 0.80, 0.95])
    b_vals = np.array([0.7, 0.65, 0.75])

    assert df.loc["A", "mean"] == pytest.approx(np.mean(a_vals))
    assert df.loc["B", "mean"] == pytest.approx(np.mean(b_vals))
    assert df.loc["A", "std"] == pytest.approx(np.std(a_vals))
    assert df.loc["B", "std"] == pytest.approx(np.std(b_vals))
    assert df.loc["A", "ci_lower"] == pytest.approx(np.percentile(a_vals, 2.5))
    assert df.loc["B", "ci_lower"] == pytest.approx(np.percentile(b_vals, 2.5))
    assert df.loc["A", "ci_upper"] == pytest.approx(np.percentile(a_vals, 97.5))
    assert df.loc["B", "ci_upper"] == pytest.approx(np.percentile(b_vals, 97.5))


def test_calculate_bootstrap_stats_ignores_groups_not_in_first_sample():
    # First sample has only A. Later samples introduce B, which should be ignored.
    samples = [
        {"A": {"Accuracy": 0.8}},  # first sample defines group set
        {"A": {"Accuracy": 0.9}, "B": {"Accuracy": 0.7}},
        {"A": {"Accuracy": 0.85}, "B": {"Accuracy": 0.75}},
    ]
    df = calculate_bootstrap_stats(samples, metric="Accuracy")
    assert set(df["group"]) == {"A"}  # B not included by design


def test_calculate_bootstrap_stats_empty_input_raises():
    with pytest.raises(IndexError):
        calculate_bootstrap_stats([], metric="Accuracy")


def test_calculate_bootstrap_stats_metric_missing_everywhere_returns_empty_df():
    # Group exists but requested metric not present in any sample
    samples = [{"A": {"Precision": 0.9}}, {"A": {"Precision": 0.8}}]
    out = calculate_bootstrap_stats(samples, metric="Accuracy")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_fast_confusion_counts_basic():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])

    tn, fp, fn, tp = metrics.fast_confusion_counts(y_true, y_pred)

    assert tp == 2
    assert tn == 2
    assert fp == 0
    assert fn == 1


def test_fast_confusion_counts_casts_to_bool():
    y_true = np.array([2, 0, -1, 0])
    y_pred = np.array([1, 0, 1, 0])

    tn, fp, fn, tp = metrics.fast_confusion_counts(y_true, y_pred)

    # After bool cast:
    # y_true -> [True, False, True, False]
    # y_pred -> [True, False, True, False]
    assert tp == 2
    assert tn == 2
    assert fp == 0
    assert fn == 0


def test_confusion_metrics_outputs_expected_keys():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    out = metrics._confusion_metrics(y_true, y_pred)

    expected_keys = {
        "TP",
        "FP",
        "FN",
        "TN",
        "TP Rate",
        "FP Rate",
        "FN Rate",
        "TN Rate",
        "Specificity",
        "Prevalence",
        "Predicted Prevalence",
    }

    assert expected_keys.issubset(out.keys())


def test_confusion_metrics_rates_are_valid():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])

    out = metrics._confusion_metrics(y_true, y_pred)

    assert out["TP"] == 0
    assert out["FN"] == 4
    assert out["TP Rate"] == 0.0
    assert out["FP Rate"] == 0.0
    assert out["Specificity"] == 0.0


def test_score_with_scorer_accuracy():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    score = metrics._score_with_scorer("accuracy", y_true, y_pred)

    assert score == pytest.approx(0.75)


def test_score_with_scorer_allows_label_based_roc_auc():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    score = metrics._score_with_scorer("roc_auc", y_true, y_pred, y_proba=None)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_with_scorer_binary_proba_matrix():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.85, 0.15],
        ]
    )


def test_get_custom_metrics_confusion_only():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    out = metrics.get_custom_metrics(
        y_true,
        y_pred,
        metric_list=["TP", "FP", "FN", "TN"],
    )

    assert out["TP"] == 1
    assert out["TN"] == 2
    assert out["FP"] == 0
    assert out["FN"] == 1


def test_get_custom_metrics_sklearn_and_confusion_mix():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    out = metrics.get_custom_metrics(
        y_true,
        y_pred,
        metric_list=["Accuracy", "TP Rate", "Specificity"],
    )

    assert "Accuracy" in out
    assert "TP Rate" in out
    assert "Specificity" in out
    assert 0.0 <= out["Accuracy"] <= 1.0


def test_calibration_auc_zero_for_perfect_calibration():
    x = np.linspace(0, 1, 10)
    auc = metrics.calibration_auc(x, x)

    assert auc == pytest.approx(0.0)


def test_calibration_auc_positive_for_miscalibration():
    mean_pred = np.array([0.2, 0.4, 0.6, 0.8])
    frac_pos = np.array([0.1, 0.3, 0.7, 0.9])

    auc = metrics.calibration_auc(mean_pred, frac_pos)

    assert auc > 0.0
