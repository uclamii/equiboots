import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.equiboots import plots


def test_save_or_show_plot(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plots.save_or_show_plot(fig, save_path=tmp_path, filename="testplot")
    assert (tmp_path / "testplot.png").exists()


def test_get_group_color_map_default():
    groups = ["A", "B", "C"]
    color_map = plots.get_group_color_map(groups)
    assert isinstance(color_map, dict)
    assert set(color_map.keys()) == set(groups)


def test_filter_groups_removes_single_class():
    data = {
        "A": {"y_true": np.array([1, 1, 1])},
        "B": {"y_true": np.array([0, 1, 0])},
    }
    valid = plots._filter_groups(data)
    assert "A" not in valid and "B" in valid


def test_filter_groups_excludes_above_threshold():
    data = {
        "A": {"y_true": np.array([0, 1])},  # 2 samples
        "B": {"y_true": np.array([0, 1, 0])},  # 3 samples
        "C": {"y_true": np.array([1])},  # 1 sample, single class (excluded)
    }
    valid = plots._filter_groups(data, exclude_groups=2)
    assert "A" in valid and "B" not in valid and "C" not in valid


def test_get_layout_default():
    n_rows, n_cols, figsize = plots.get_layout(10)
    assert n_rows > 0 and n_cols > 0
    assert isinstance(figsize, tuple)


def test_get_layout_custom_figsize():
    _, _, figsize = plots.get_layout(4, n_cols=2, figsize=(12, 8))
    assert figsize == (12, 8)


def test_plot_with_layout_group_not_found(monkeypatch):
    data = {"A": {"y_true": np.array([1, 1]), "y_pred": np.array([1, 1])}}
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.plot_with_layout(data, lambda *a, **k: None, {}, group="B")


def test_plot_with_layout_overlay(monkeypatch):
    data = {
        "A": {"y_true": np.array([0, 1]), "y_pred": np.array([0.2, 0.8])},
        "B": {"y_true": np.array([1, 0]), "y_pred": np.array([0.9, 0.1])},
    }
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.plot_with_layout(
        data,
        lambda ax, data, group, color, **kwargs: ax.plot([0, 1], [0, 1]),
        {},
        subplots=False,
    )


def test_plot_with_layout_subplots(monkeypatch):
    data = {
        "A": {"y_true": np.array([0, 1]), "y_pred": np.array([0.2, 0.8])},
        "B": {"y_true": np.array([1, 0]), "y_pred": np.array([0.9, 0.1])},
    }
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.plot_with_layout(
        data,
        lambda ax, data, group, color, **kwargs: ax.plot([0, 1], [0, 1]),
        {},
        subplots=True,
    )


def test_eq_plot_residuals_by_group_overlay(monkeypatch):
    data = {
        "A": {"y_true": np.array([3, 2, 4]), "y_pred": np.array([2.5, 2, 3.5])},
        "B": {"y_true": np.array([1, 0, 2]), "y_pred": np.array([1, 0.5, 2])},
    }
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_residuals_by_group(data, subplots=False)


def test_eq_plot_residuals_by_group_specific_group(monkeypatch):
    data = {
        "A": {"y_true": np.array([3, 2, 4]), "y_pred": np.array([2.5, 2, 3.5])},
    }
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_residuals_by_group(data, group="A")


def test_eq_plot_residuals_by_group_invalid_group(monkeypatch):
    data = {
        "A": {"y_true": np.array([3, 2, 4]), "y_pred": np.array([2.5, 2, 3.5])},
    }
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_residuals_by_group(data, group="B")


def test_eq_plot_group_curves_invalid_group(monkeypatch):
    data = {
        "A": {"y_true": np.array([1, 1, 1]), "y_prob": np.array([0.9, 0.8, 0.95])},
    }
    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.raises(Exception, match="No members in group below"):
        plots.eq_plot_group_curves(data, curve_type="roc")


def test_interpolate_bootstrapped_curves_invalid(monkeypatch):
    data = [{"A": {"y_true": np.array([1, 1]), "y_prob": np.array([0.9, 0.8])}}]
    grid = np.linspace(0, 1, 10)
    result, gx = plots.interpolate_bootstrapped_curves(
        data, grid, curve_type="calibration"
    )
    assert isinstance(result, dict) and gx is not None


def test_eq_plot_bootstrapped_group_curves_overlay(monkeypatch):
    boot_data = [
        {
            "A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.1, 0.9])},
            "B": {"y_true": np.array([1, 0]), "y_prob": np.array([0.8, 0.2])},
        },
        {
            "A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.2, 0.8])},
            "B": {"y_true": np.array([1, 0]), "y_prob": np.array([0.9, 0.1])},
        },
    ]
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_bootstrapped_group_curves(boot_data, curve_type="roc", subplots=False)


def test_eq_plot_bootstrapped_group_curves_subplot(monkeypatch):
    boot_data = [
        {
            "A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.1, 0.9])},
        },
        {
            "A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.2, 0.8])},
        },
    ]
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_bootstrapped_group_curves(boot_data, curve_type="roc", subplots=True)


def test_eq_plot_bootstrapped_group_curves_specific_group(monkeypatch):
    boot_data = [
        {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.1, 0.9])}},
        {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.2, 0.8])}},
    ]
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_bootstrapped_group_curves(boot_data, curve_type="roc", group="A")


def test_eq_plot_group_curves_all_curve_types(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    data = {
        "A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.1, 0.9])},
        "B": {"y_true": np.array([1, 0]), "y_prob": np.array([0.8, 0.2])},
    }
    for curve in ["roc", "pr", "calibration"]:
        plots.eq_plot_group_curves(data, curve_type=curve)


def test_eq_disparity_metrics_plot_typeerror():
    with pytest.raises(TypeError):
        plots.eq_disparity_metrics_plot(
            dispa={"A": {"Accuracy_ratio": 1.0}},
            metric_cols=["Accuracy_ratio"],
            name="test",
        )


def test_eq_disparity_metrics_plot_invalid_kind():
    dispa = [
        {"A": {"Metric1": 0.9}},
        {"A": {"Metric1": 1.1}},
    ]
    with pytest.raises(ValueError):
        plots.eq_disparity_metrics_plot(
            dispa=dispa,
            metric_cols=["Metric1"],
            name="group",
            plot_kind="invalid_plot_type",
        )
