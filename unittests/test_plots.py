import numpy as np
import pytest
import pandas as pd
import matplotlib.pyplot as plt

from equiboots.plots import (
    eq_plot_roc_auc,
    eq_plot_precision_recall,
    eq_calibration_curve_plot,
    eq_disparity_metrics_plot,
)

# --- Fixtures --- #


@pytest.fixture
def synthetic_fairness_data():
    y_prob = np.random.rand(100)
    y_true = np.random.randint(0, 2, 100)
    group_labels = np.random.choice(["A", "B"], 100)
    data = {
        group: {
            "y_true": y_true[group_labels == group],
            "y_prob": y_prob[group_labels == group],
        }
        for group in np.unique(group_labels)
    }
    return data


# --- Smoke tests (no return check) --- #


def test_eq_plot_roc_auc_runs(synthetic_fairness_data):
    eq_plot_roc_auc(data=synthetic_fairness_data)  # Just run, ensure no error
    plt.close("all")


def test_eq_plot_precision_recall_runs(synthetic_fairness_data):
    eq_plot_precision_recall(data=synthetic_fairness_data)
    plt.close("all")


def test_eq_calibration_curve_plot_runs(synthetic_fairness_data):
    eq_calibration_curve_plot(data=synthetic_fairness_data)
    plt.close("all")


def test_eq_disparity_metrics_plot_runs():
    dispa = [
        {
            "A": {"Accuracy_ratio": 1.05, "Precision_ratio": 0.95},
            "B": {"Accuracy_ratio": 0.95, "Precision_ratio": 1.10},
        }
    ]
    eq_disparity_metrics_plot(
        dispa=dispa,
        metric_cols=["Accuracy_ratio", "Precision_ratio"],
        name="race",
        categories="all",
        plot_kind="violinplot",
    )
    plt.close("all")
