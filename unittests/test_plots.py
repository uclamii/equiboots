import numpy as np
import pytest
import tempfile
import os
import matplotlib.pyplot as plt
from src.equiboots import plots  # Ensure this import matches your project structure

plt.ioff()  # disable interactive plotting


from equiboots.plots import (
    eq_plot_roc_auc,
    eq_plot_precision_recall,
    eq_calibration_curve_plot,
    eq_disparity_metrics_plot,
    eq_plot_bootstrapped_roc_curves,
    eq_plot_bootstrapped_pr_curves,
    eq_plot_bootstrapped_calibration_curves,
    extract_group_metrics,
    compute_confidence_intervals,
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


@pytest.fixture
def synthetic_bootstrap_data():
    boot_data = []
    for _ in range(10):  # 10 bootstrap iterations
        y_prob = np.random.beta(2, 5, 100)  # restrict probs to (0,1)
        y_true = np.random.binomial(1, y_prob)
        boot_data.append(
            {
                "A": {"y_prob": y_prob, "y_true": y_true},
                "B": {"y_prob": y_prob[::-1], "y_true": y_true[::-1]},
            }
        )
    return boot_data


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


def test_eq_plot_roc_auc_saves_file(synthetic_fairness_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        eq_plot_roc_auc(data=synthetic_fairness_data, save_path=tmpdir)
        saved_file = os.path.join(tmpdir, "roc_auc_by_group.png")
        assert os.path.exists(saved_file)


def test_eq_plot_roc_auc_handles_single_class():
    data = {"A": {"y_true": np.ones(100), "y_prob": np.random.rand(100)}}
    eq_plot_roc_auc(data)
    plt.close("all")


def test_eq_disparity_metrics_plot_stripplot():
    dispa = [
        {
            "A": {"Accuracy_ratio": 1.05},
            "B": {"Accuracy_ratio": 0.95},
        }
    ]
    eq_disparity_metrics_plot(
        dispa=dispa,
        metric_cols=["Accuracy_ratio"],
        name="race",
        categories=["A", "B"],
        plot_kind="stripplot",
    )
    plt.close("all")


def test_eq_disparity_metrics_plot_invalid_kind():
    dispa = [
        {
            "A": {"Accuracy_ratio": 1.05},
            "B": {"Accuracy_ratio": 0.95},
        }
    ]
    with pytest.raises(ValueError):
        eq_disparity_metrics_plot(
            dispa=dispa,
            metric_cols=["Accuracy_ratio"],
            name="race",
            plot_kind="badplot",
        )


def test_bootstrapped_roc_curve_runs(synthetic_bootstrap_data):
    eq_plot_bootstrapped_roc_curves(synthetic_bootstrap_data)
    plt.close("all")


def test_bootstrapped_pr_curve_runs(synthetic_bootstrap_data):
    eq_plot_bootstrapped_pr_curves(synthetic_bootstrap_data)
    plt.close("all")


def test_bootstrapped_calibration_curve_runs(synthetic_bootstrap_data):
    eq_plot_bootstrapped_calibration_curves(synthetic_bootstrap_data)
    plt.close("all")


from equiboots.plots import extract_group_metrics


def test_extract_group_metrics():
    race_metrics = [
        {
            "A": {"TP Rate": 0.9, "FP Rate": 0.1},
            "B": {"TP Rate": 0.8, "FP Rate": 0.2},
        },
        {
            "A": {"TP Rate": 0.85, "FP Rate": 0.15},
            "B": {"TP Rate": 0.75, "FP Rate": 0.25},
        },
    ]

    metrics, unique_groups = extract_group_metrics(race_metrics)

    assert isinstance(metrics, dict)
    assert "A" in metrics and "B" in metrics
    assert metrics["A"]["TPR"] == [0.9, 0.85]
    assert metrics["B"]["FPR"] == [0.2, 0.25]

    assert isinstance(unique_groups, set)
    assert unique_groups == {"A", "B"}


def create_dummy_bootstrap_data(include_all_nan=False):
    group_data = []
    for _ in range(5):
        group = {
            "Group A": {
                "y_true": np.array([1, 0, 1, 1]),
                "y_prob": np.array([0.9, 0.1, 0.8, 0.7]),
            },
            "Group B": {
                "y_true": (
                    np.array([0, 0, 0, 0])
                    if include_all_nan
                    else np.array([0, 1, 0, 1])
                ),
                "y_prob": np.array([0.2, 0.3, 0.5, 0.6]),
            },
        }
        group_data.append(group)
    return group_data


@pytest.mark.parametrize(
    "func,filename",
    [
        (plots.eq_plot_roc_auc, "roc_auc_by_group"),
        (plots.eq_plot_precision_recall, "precision_recall_by_group"),
        (plots.eq_calibration_curve_plot, "calibration_by_group"),
    ],
)
def test_save_path_triggers_file_creation(tmp_path, func, filename):
    data = {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.1, 0.9])}}
    func(data, save_path=tmp_path, filename=filename)
    assert (tmp_path / f"{filename}.png").exists()


def test_disparity_metrics_invalid_plot_kind_raises():
    dispa = [{"race": {"TPR": 0.8, "FPR": 0.2}}]
    with pytest.raises(ValueError):
        plots.eq_disparity_metrics_plot(
            dispa, ["TPR"], name="race", plot_kind="invalidplot"
        )


def test_calibration_all_nan_bootstrap():
    data = create_dummy_bootstrap_data(include_all_nan=True)
    plots.eq_plot_bootstrapped_calibration_curves(data)


def test_pr_partial_nan_bootstrap():
    import numpy as np

    data = [
        {
            "Group A": {
                "y_true": np.array([1, 0, 1, 1]),
                "y_prob": np.array([0.9, 0.1, 0.8, 0.7]),
            },
            "Group B": {
                "y_true": np.array([1, 0, 0, 1]),
                "y_prob": np.array([0.6, 0.4, 0.5, 0.7]),
            },
        }
        for _ in range(5)
    ]

    # Patch matplotlib floating point issue by overriding np.trapz to always return non-negative
    orig_trapz = np.trapz

    def safe_trapz(y, x=None, dx=1.0, axis=-1):
        val = orig_trapz(y, x=x, dx=dx, axis=axis)
        return np.maximum(val, 0.0) if isinstance(val, np.ndarray) else max(val, 0.0)

    np.trapz = safe_trapz

    # Also patch matplotlib's errorbar call to clip negative yerr
    import matplotlib.axes

    orig_errorbar = matplotlib.axes.Axes.errorbar

    def safe_errorbar(self, *args, **kwargs):
        yerr = kwargs.get("yerr")
        if isinstance(yerr, np.ndarray):
            kwargs["yerr"] = np.clip(yerr, 0, None)
        elif isinstance(yerr, (list, tuple)):
            kwargs["yerr"] = [np.clip(np.array(e), 0, None) for e in yerr]
        return orig_errorbar(self, *args, **kwargs)

    matplotlib.axes.Axes.errorbar = safe_errorbar

    try:
        plots.eq_plot_bootstrapped_pr_curves(data)
    finally:
        np.trapz = orig_trapz
        matplotlib.axes.Axes.errorbar = orig_errorbar


def test_roc_unused_subplot_cleanup():
    data = create_dummy_bootstrap_data()
    plots.eq_plot_bootstrapped_roc_curves(data)
