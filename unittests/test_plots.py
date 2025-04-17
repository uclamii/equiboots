import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.equiboots import plots


def test_save_or_show_plot(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.save_or_show_plot(fig, save_path=str(tmp_path), filename="test_plot")
    assert (tmp_path / "test_plot.png").exists()


def test_get_group_color_map():
    groups = ["A", "B", "C"]
    cmap = plots.get_group_color_map(groups)
    assert set(cmap.keys()) == set(groups)
    assert all(isinstance(color, tuple) for color in cmap.values())


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
    plots.eq_plot_bootstrapped_group_curves(
        boot_data,
        curve_type="roc",
        subplots=False,
    )


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
    plots.eq_plot_bootstrapped_group_curves(
        boot_data,
        curve_type="roc",
        subplots=True,
    )


def test_eq_plot_bootstrapped_group_curves_specific_group(monkeypatch):
    boot_data = [
        {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.1, 0.9])}},
        {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.2, 0.8])}},
    ]
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_bootstrapped_group_curves(
        boot_data,
        curve_type="roc",
        group="A",
    )


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


def test_filter_groups_excludes_named_groups():
    data = {
        "A": {"y_true": np.array([0, 1])},
        "B": {"y_true": np.array([1, 0])},
        "C": {"y_true": np.array([1, 0])},
    }

    out1 = plots._filter_groups(data, exclude_groups="B")  # Single string
    assert "B" not in out1 and "A" in out1 and "C" in out1

    out2 = plots._filter_groups(data, exclude_groups=["A", "C"])  # List
    assert "A" not in out2 and "C" not in out2 and "B" in out2

    out3 = plots._filter_groups(data, exclude_groups={"A"})  # Set
    assert "A" not in out3 and "B" in out3 and "C" in out3


def test_validate_plot_kwargs_errors():
    with pytest.raises(ValueError, match="must be a dictionary"):
        plots._validate_plot_kwargs(["not", "a", "dict"])

    with pytest.raises(ValueError, match="invalid group names"):
        plots._validate_plot_kwargs(
            {"Z": {"color": "red"}},
            valid_groups=["A", "B"],
        )

    with pytest.raises(ValueError, match="must be a dictionary"):
        plots._validate_plot_kwargs({"A": "not-a-dict"}, valid_groups=["A"])

    with pytest.raises(ValueError, match="contains invalid plot arguments"):
        plots._validate_plot_kwargs(
            {"A": {"banana": "yellow"}},
            valid_groups=["A"],
        )


def test_plot_with_layout_raises_on_empty_groups(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.raises(Exception, match="No members in group below"):
        plots.plot_with_layout({}, lambda *a, **k: None, {})


def test_get_group_color_map_custom_palette():
    groups = ["X", "Y"]
    cmap = plots.get_group_color_map(groups, palette="tab20")
    assert isinstance(cmap, dict)
    assert set(cmap.keys()) == set(groups)


def test_filter_groups_invalid_type():
    with pytest.raises(
        ValueError, match="exclude_groups must be an int, str, list, or set"
    ):
        plots._filter_groups(
            {"A": {"y_true": np.array([0, 1])}},
            exclude_groups=3.5,
        )


def test_plot_with_layout_warns_on_invalid_group(monkeypatch):
    data = {"A": {"y_true": np.array([0, 1]), "y_pred": np.array([0.2, 0.8])}}
    monkeypatch.setattr(plt, "show", lambda: None)
    # Should trigger print warning and exit early
    plots.plot_with_layout(data, lambda *a, **k: None, {}, group="B")


def test_plot_with_layout_no_grid(monkeypatch):
    data = {"A": {"y_true": np.array([0, 1]), "y_pred": np.array([0.2, 0.8])}}
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.plot_with_layout(
        data,
        lambda ax, d, g, c, **kw: ax.plot([0, 1], [0, 1]),
        {},
        show_grid=False,
    )


def test_eq_plot_group_curves_pr(monkeypatch):
    data = {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.3, 0.9])}}
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_group_curves(data, curve_type="pr")


def test_eq_plot_group_curves_calibration(monkeypatch):
    data = {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.3, 0.9])}}
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_group_curves(data, curve_type="calibration")


def test_eq_plot_bootstrapped_group_curves_calibration(monkeypatch):
    boot_data = [
        {
            "A": {
                "y_true": np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1]),
                "y_prob": np.array(
                    [0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
                ),
            }
        },
        {
            "A": {
                "y_true": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
                "y_prob": np.array(
                    [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                ),
            }
        },
    ]
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_bootstrapped_group_curves(boot_data, curve_type="calibration")


def test_eq_disparity_metrics_plot_violin(monkeypatch):
    dispa = [
        {"A": {"Metric1": 0.9, "Metric2": 0.1}},
        {"A": {"Metric1": 1.1, "Metric2": 0.2}},
    ]
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_disparity_metrics_plot(
        dispa,
        metric_cols=["Metric1", "Metric2"],
        name="test",
        plot_kind="violinplot",
        categories=["A"],
    )


def test_validate_plot_data_nan():
    data = {"A": {"y_true": np.array([1, np.nan]), "y_prob": np.array([0.5, 0.5])}}
    with pytest.raises(ValueError):
        plots._validate_plot_data(data)


def test_validate_plot_kwargs_invalid_group():
    plot_kwargs = {"Z": {"color": "red"}}
    with pytest.raises(ValueError):
        plots._validate_plot_kwargs(plot_kwargs, valid_groups=["A", "B"])


def test_validate_plot_data_raises_missing_keys():
    bad_data = {"A": {"y_prob": np.array([0.1, 0.2])}}  # Missing y_true
    with pytest.raises(ValueError, match="y_true missing for group 'A'"):
        plots._validate_plot_data(bad_data)


def test_validate_plot_data_raises_nans():
    bad_data = {"A": {"y_true": np.array([0, np.nan]), "y_prob": np.array([0.1, 0.2])}}
    with pytest.raises(ValueError, match="NaN values found in y_true"):
        plots._validate_plot_data(bad_data)


def test_concatenated_group_data():
    boot_data = [
        {"A": {"y_true": np.array([1]), "y_prob": np.array([0.5])}},
        {"A": {"y_true": np.array([0]), "y_prob": np.array([0.3])}},
    ]
    out = plots._get_concatenated_group_data(boot_data)
    assert np.array_equal(out["A"]["y_true"], np.array([1, 0]))
    assert np.array_equal(out["A"]["y_prob"], np.array([0.5, 0.3]))


def test_bootstrapped_calibration_with_brier(monkeypatch):
    # Use more samples and ensure y_prob values span the full [0,1] range
    boot_data = [
        {
            "A": {
                "y_true": np.random.randint(0, 2, 100),
                "y_prob": np.linspace(0.01, 0.99, 100),
            }
        },
        {
            "A": {
                "y_true": np.random.randint(0, 2, 100),
                "y_prob": np.linspace(0.01, 0.99, 100)[::-1],
            }
        },
    ]
    monkeypatch.setattr(plt, "show", lambda: None)
    plots.eq_plot_bootstrapped_group_curves(boot_data, curve_type="calibration")


def test_eq_plot_group_curves_invalid_curve_type():
    data = {"A": {"y_true": np.array([1, 0]), "y_prob": np.array([0.6, 0.4])}}
    with pytest.raises(ValueError, match="Unsupported curve_type"):
        plots.eq_plot_group_curves(data, curve_type="unsupported")


def test_plot_group_curve_ax_all_labels(monkeypatch):
    # Covers AUC and Brier score logic
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    data = {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.3, 0.9])}}
    plots._plot_group_curve_ax(
        ax,
        data,
        group="A",
        color="blue",
        curve_type="roc",
        label_mode="full",
        is_subplot=True,
        single_group=True,
    )


def test_plot_bootstrapped_curve_ax_with_limits(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    y_array = np.array([[0.1 * i for i in range(10)] for _ in range(100)])
    grid_x = np.linspace(0, 1, 10)
    plots._plot_bootstrapped_curve_ax(
        ax,
        y_array,
        grid_x,
        group="A",
        label_prefix="AUROC",
        y_lim=(0, 1),
    )


def test_filter_groups_exclude_by_name():
    data = {
        "A": {"y_true": np.array([0, 1])},
        "B": {"y_true": np.array([0, 0])},
    }
    filtered = plots._filter_groups(data, exclude_groups=["B"])
    assert "A" in filtered
    assert "B" not in filtered


def test_eq_disparity_metrics_point_plot_divide_by_zero(monkeypatch):
    """Ensure plot handles divide-by-zero gracefully when reference metric is zero."""
    # Suppress plot display during test execution
    monkeypatch.setattr(plt, "show", lambda: None)

    # Simulate disparity input where the reference group has a zero metric value
    dispa = [
        {
            "White": {"Accuracy": 0.0},  # Reference group with zero value
            "Black": {"Accuracy": 0.9},
        }
    ]

    # Verify that plotting function executes without error
    plots.eq_disparity_metrics_point_plot(
        dispa,
        metric_cols=["Accuracy"],
        category_names=["Race"],
    )


def test_filter_groups_exclude_invalid_type():
    data = {"A": {"y_true": np.array([0, 1])}}
    with pytest.raises(ValueError):
        plots._filter_groups(data, exclude_groups=3.14)


def test_filter_groups_include_all():
    data = {
        "A": {"y_true": np.array([0, 1])},
        "B": {"y_true": np.array([1, 1])},
    }
    result = plots._filter_groups(data, exclude_groups=0)
    assert "A" in result
    assert "B" not in result  # single class


def test_plot_group_curve_ax_with_title(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    data = {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.2, 0.8])}}
    plots._plot_group_curve_ax(
        ax, data, "A", color="blue", curve_type="roc", title="Test Title"
    )
    assert ax.get_title() == "Test Title"


def test_plot_group_curve_ax_tick_params(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    data = {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.2, 0.8])}}
    plots._plot_group_curve_ax(ax, data, "A", color="green", curve_type="roc")
    ticks = ax.xaxis.get_tick_params()
    assert ticks is not None


def test_plot_with_layout_default_legend(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    data = {
        "A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.2, 0.8])},
        "B": {"y_true": np.array([1, 0]), "y_prob": np.array([0.8, 0.2])},
    }

    def safe_plot(ax, d, g, c, **kw):
        # Remove any keys that aren't valid for ax.plot
        kw.pop("overlay_mode", None)
        ax.plot(d[g]["y_prob"], d[g]["y_true"], color=c, label=g, **kw)

    plots.plot_with_layout(
        data,
        safe_plot,
        {},
        title="Demo",
        subplots=False,
    )


def test_validate_plot_kwargs_invalid_type():
    with pytest.raises(ValueError):
        plots._validate_plot_kwargs(plot_kwargs="invalid")


def test_validate_plot_kwargs_invalid_plot_arg():
    with pytest.raises(ValueError):
        plots._validate_plot_kwargs({"color": "blue", "invalid_kwarg": 123})


def test_interpolate_bootstrapped_curves_empty():
    boot_sliced_data = [
        {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.1, 0.9])}}
    ]
    grid_x = np.linspace(0, 1, 10)
    out, _ = plots.interpolate_bootstrapped_curves(
        boot_sliced_data, grid_x, curve_type="roc"
    )
    assert "A" in out
    assert isinstance(out["A"], list)


def test_validate_plot_data_bootstrap():
    data = [
        {"A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.5, 0.6])}},
        {"A": {"y_true": np.array([1, 0]), "y_prob": np.array([0.6, 0.4])}},
    ]
    # This should pass without error
    plots._validate_plot_data(data, is_bootstrap=True)


def test_validate_plot_kwargs_group_level_invalid():
    with pytest.raises(ValueError, match="contains invalid plot arguments"):
        plots._validate_plot_kwargs({"A": {"invalid_kwarg": 123}}, valid_groups=["A"])


def test_overlay_mode_triggers_clear(monkeypatch):
    class MockFig:
        def tight_layout(self, *args, **kwargs):
            pass

        def suptitle(self, *args, **kwargs):
            pass

    class MockAx:
        def __init__(self):
            self.cleared = False
            self.logged = []

        def clear(self):
            self.cleared = True

        def plot(self, *args, **kwargs):
            self.logged.append(("plot", args, kwargs))

        def set_title(self, *args):
            self.logged.append(("title", args))

        def set_xlabel(self, *args, **kwargs):
            self.logged.append(("xlabel", args, kwargs))

        def set_ylabel(self, *args, **kwargs):
            self.logged.append(("ylabel", args, kwargs))

        def legend(self, *args, **kwargs):
            self.logged.append(("legend", args, kwargs))

        def grid(self, *args, **kwargs):
            self.logged.append(("grid", args, kwargs))

        def tick_params(self, *args, **kwargs):
            self.logged.append(("tick_params", args, kwargs))

        def axis(self, *args, **kwargs):
            self.logged.append(("axis", args, kwargs))

    fig = MockFig()
    ax = MockAx()
    monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fig, np.array([ax])))
    monkeypatch.setattr(plt, "show", lambda: None)

    data = {
        "A": {"y_true": np.array([0, 1]), "y_prob": np.array([0.3, 0.7])},
        "B": {"y_true": np.array([1, 0]), "y_prob": np.array([0.6, 0.4])},
    }

    def dummy_func(ax, data, group, color, overlay_mode=False, *args, **kwargs):
        y_true = data[group]["y_true"]
        y_prob = data[group]["y_prob"]
        ax.plot(y_prob, y_true, color=color, *args, **kwargs)
        if not overlay_mode:
            ax.clear()

    plots.plot_with_layout(data, dummy_func, {}, subplots=True, n_cols=2)
    assert ax.cleared is True
