import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import seaborn as sns
from packaging import version
from .metrics import regression_metrics, calibration_auc, calculate_bootstrap_stats
from typing import Dict, List, Optional, Union, Tuple, Set, Callable


SEABORN_OLD = version.parse(sns.__version__) < version.parse("0.13.2")

################################################################################
# Shared Utilities
################################################################################

DEFAULT_LINE_KWARGS = {"color": "black", "linestyle": "--", "linewidth": 1}
DEFAULT_LEGEND_KWARGS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, -0.25),
    "ncol": 1,
}

VALID_PLOT_KWARGS = {
    "color",
    "linestyle",
    "linewidth",
    "marker",
    "markersize",
    "alpha",
    "markeredgecolor",
    "markeredgewidth",
    "markerfacecolor",
    "dash_capstyle",
    "dash_joinstyle",
    "solid_capstyle",
    "solid_joinstyle",
    "zorder",
}


def save_or_show_plot(
    fig: plt.Figure,
    save_path: Optional[str] = None,
    filename: str = "plot",
) -> None:
    """Save plot to file if path is provided, otherwise display it."""

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename}.png"),
            bbox_inches="tight",
        )
    plt.show()


def get_group_color_map(
    groups: List[str],
    palette: str = "tab10",
) -> Dict[str, str]:
    """Generate a mapping from group names to colors."""

    colors = plt.get_cmap(palette).colors
    return {g: colors[i % len(colors)] for i, g in enumerate(groups)}


def get_layout(
    n_items: int,
    n_cols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    strict_layout: bool = True,
) -> Tuple[int, int, Tuple[float, float]]:
    """Compute layout grid and figure size based on number of items."""

    n_cols = n_cols or (6 if strict_layout else int(np.ceil(np.sqrt(n_items))))
    n_rows = int(np.ceil(n_items / n_cols))
    # Check if the grid is sufficient to hold all items
    if n_rows * n_cols < n_items:
        raise ValueError(
            f"Subplot grid is too small: {n_rows} rows * {n_cols} cols = "
            f"{n_rows * n_cols} slots, but {n_items} items need to be plotted. "
            f"Increase `n_cols` or allow more rows."
        )
    fig_width, fig_height = figsize or (
        (24, 4 * n_rows) if strict_layout else (5 * n_cols, 5 * n_rows)
    )
    return n_rows, n_cols, (fig_width, fig_height)


def _filter_groups(
    data: Dict[str, Dict[str, np.ndarray]],
    exclude_groups: Union[int, str, List[str], Set[str]] = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Filter out groups with one class or based on exclusion criteria."""

    valid_data = {g: v for g, v in data.items() if len(set(v["y_true"])) > 1}
    if not exclude_groups:  # If exclude_groups is 0 or None, return all valid data
        return valid_data

    # Handle case where exclude_groups is specific group name (str) or list of names
    if isinstance(exclude_groups, (str, list, set)):
        exclude_set = (
            {exclude_groups} if isinstance(exclude_groups, str) else set(exclude_groups)
        )
        return {g: v for g, v in valid_data.items() if g not in exclude_set}

    # Handle case where exclude_groups is an integer (minimum sample size threshold)
    if isinstance(exclude_groups, int):
        return {
            g: v for g, v in valid_data.items() if len(v["y_true"]) <= exclude_groups
        }

    raise ValueError("exclude_groups must be an int, str, list, or set")


def _get_concatenated_group_data(
    boot_sliced_data: List[Dict[str, Dict[str, np.ndarray]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Concatenate bootstrapped data across samples."""
    return {
        g: {
            "y_true": np.concatenate(
                [bs[g]["y_true"] for bs in boot_sliced_data if g in bs]
            ),
            "y_prob": np.concatenate(
                [bs[g]["y_prob"] for bs in boot_sliced_data if g in bs]
            ),
        }
        for g in set(g for bs in boot_sliced_data for g in bs)
    }


def _validate_plot_data(
    data: Union[
        Dict[str, Dict[str, np.ndarray]],
        List[Dict[str, Dict[str, np.ndarray]]],
    ],
    is_bootstrap: bool = False,
) -> None:
    """
    Validate plot data for missing y_true/y_prob (or y_pred) and NaN values.
    """

    # Convert single dict to a list of one dict for unified processing
    data_iter = data if is_bootstrap else [data]
    context = " in bootstrap sample" if is_bootstrap else ""

    for d in data_iter:
        for g, values in d.items():
            # Check for missing keys
            y_true = values.get("y_true", values.get("y_actual"))
            y_prob = values.get("y_prob", values.get("y_pred"))
            if y_true is None:
                raise ValueError(f"y_true missing for group '{g}'{context}")
            if y_prob is None:
                raise ValueError(f"y_prob missing for group '{g}'{context}")
            # Check for NaN values
            if np.any(np.isnan(y_true)):
                raise ValueError(f"NaN values found in y_true for group '{g}'{context}")
            if np.any(np.isnan(y_prob)):
                raise ValueError(f"NaN values found in y_prob for group '{g}'{context}")


def _validate_plot_kwargs(
    plot_kwargs: Optional[Dict[str, Union[Dict[str, str], Dict[str, float]]]],
    valid_groups: Optional[List[str]] = None,
    kwarg_name: str = "plot_kwargs",
) -> None:
    """Validate keyword arguments for use in Matplotlib's plot function."""

    if plot_kwargs is None:
        return

    if not isinstance(plot_kwargs, dict):
        raise ValueError(f"{kwarg_name} must be a dictionary, got {type(plot_kwargs)}")

    # If valid_groups is provided, plot_kwargs maps groups to kwargs (curve_kwgs case)
    if valid_groups is not None:
        # Check for invalid group names
        invalid_groups = set(plot_kwargs.keys()) - set(valid_groups)
        if invalid_groups:
            raise ValueError(
                f"{kwarg_name} contains invalid group names: {invalid_groups}"
            )

        # Validate each group's kwargs
        for group, kwargs in plot_kwargs.items():
            if not isinstance(kwargs, dict):
                raise ValueError(
                    f"{kwarg_name} for group '{group}' must be a dictionary, "
                    f"got {type(kwargs)}"
                )
            # Check for invalid kwargs
            invalid_kwargs = set(kwargs.keys()) - VALID_PLOT_KWARGS
            if invalid_kwargs:
                raise ValueError(
                    f"{kwarg_name} for group '{group}' contains invalid plot "
                    f"arguments: {invalid_kwargs}. "
                    f"Valid arguments are: {VALID_PLOT_KWARGS}"
                )
    # If `valid_groups` is `None`, `plot_kwargs` is a single dict of kwargs
    else:
        # Check for invalid kwargs
        invalid_kwargs = set(plot_kwargs.keys()) - VALID_PLOT_KWARGS
        if invalid_kwargs:
            raise ValueError(
                f"{kwarg_name} contains invalid plot arguments: {invalid_kwargs}. "
                f"Valid arguments are: {VALID_PLOT_KWARGS}"
            )


def plot_with_layout(
    data: Union[
        Dict[str, Dict[str, np.ndarray]], List[Dict[str, Dict[str, np.ndarray]]]
    ],
    plot_func: Callable,
    plot_kwargs: Dict,
    title: str = "Plot",
    filename: str = "plot",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    subplots: bool = False,
    n_cols: int = 2,
    n_rows: Optional[int] = None,
    group: Optional[str] = None,
    color_by_group: bool = True,
    exclude_groups: Union[int, str, List[str], Set[str]] = 0,
    show_grid: bool = True,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Master plotting wrapper that handles 3 layout modes:
    1. Single group plot (if group is passed)
    2. Subplots mode: one axis per group
    3. Overlay mode: all groups on one axis

    plot_func : callable
        Function of signature (ax, data, group_name, color, **kwargs)
        Must handle a `overlay_mode` kwarg to distinguish plot logic.
    """

    valid_data = data
    groups = sorted(valid_data.keys())
    if len(groups) == 0:
        raise Exception(f"No members in group below {exclude_groups}.")
    color_map = (
        get_group_color_map(groups)
        if color_by_group
        else {g: "#1f77b4" for g in groups}
    )

    if group:
        if group not in valid_data:
            print(f"[Warning] Group '{group}' not found.")
            return
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plot_func(
            ax,
            valid_data,
            group,
            color_map[group],
            **plot_kwargs,
            overlay_mode=False,
        )
        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.set_title(f"{title} ({group})")
        fig.tight_layout()
        save_or_show_plot(fig, save_path, f"{filename}_{group}")
        return

    if subplots:
        n_rows = n_rows or int(np.ceil(len(groups) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
            dpi=dpi,
        )
        axes = axes.flatten()
        for i, g in enumerate(groups):
            if i >= len(axes):
                break
            plot_func(
                axes[i],
                valid_data,
                g,
                color_map[g],
                **plot_kwargs,
                overlay_mode=False,
            )
            if y_lim is not None:
                axes[i].set_ylim(y_lim)
        for j in range(i + 1, len(axes)):  # Hide unused subplots
            axes[j].axis("off")

        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:  # ---- Mode 3: overlay
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for g in groups:
            plot_func(
                ax,
                valid_data,
                g,
                color_map[g],
                **plot_kwargs,
                overlay_mode=True,
            )
        ax.set_title(title)
        ax.legend(**DEFAULT_LEGEND_KWARGS)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if y_lim is not None:
            ax.set_ylim(y_lim)

    if show_grid:
        plt.grid(linestyle=":")

    save_or_show_plot(fig, save_path, filename)


def add_plot_threshold_lines(
    ax: plt.Axes,
    lower: float,
    upper: float,
    xmax: float,
    show_reference: bool = True,
) -> None:
    """Add threshold lines to the plot, optionally showing y=1 reference line."""
    y_values = [lower, upper]
    colors = ["red", "red"]
    if show_reference:
        y_values.insert(1, 1.0)
        colors.insert(1, "black")
    ax.hlines(y_values, xmin=-0.5, xmax=xmax + 0.5, ls=":", colors=colors)
    ax.set_xlim(-0.5, xmax + 0.5)


def generate_alpha_labels(n: int) -> List[str]:
    """Generate alphabetical labels for n groups (A, B, ..., Z, AA, AB, ...)."""
    labels = []
    for i in range(n):
        if i < 26:
            # Single letter: A to Z
            labels.append(chr(65 + i))
        else:
            # Double letter: AA, AB, ..., AZ, BA, BB, ...
            first = chr(65 + (i // 26 - 1))  # First letter (A, B, ...)
            second = chr(65 + (i % 26))  # Second letter (A, B, ...)
            labels.append(first + second)
    return labels


def setup_plot_environment(
    group_list: List[str],
    cmap: str,
    color_by_group: bool,
    metric_cols: List[str],
    max_cols: Optional[int],
    n_cols: int,
    figsize: Optional[Tuple[float, float]],
    strict_layout: bool,
    y_lim: Optional[Tuple[float, float]] = None,
    layout_type: str = "point",
) -> Tuple[
    plt.Figure,
    np.ndarray,
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    int,
    int,
]:
    """Set up colors, labels, and subplot grid for plotting."""
    # Color setup
    color_map = plt.get_cmap(cmap)
    colors = [color_map(i / len(group_list)) for i in range(len(group_list))]
    base_colors = {
        group: colors[i] if color_by_group else "#1f77b4"
        for i, group in enumerate(group_list)
    }

    # Alphabetical labels
    alpha_labels = generate_alpha_labels(len(group_list))
    group_to_alpha = dict(zip(group_list, alpha_labels))
    alpha_to_group = dict(zip(alpha_labels, group_list))

    # Subplot grid setup
    if layout_type == "point":
        # For eq_group_metrics_point_plot: one row per metric, one col. per cat.
        n_rows = len(metric_cols)
        n_cols = n_cols  # Already set to len(category_names)
    else:
        # For eq_group_metrics_plot: one column per metric, calculate rows
        n_cols = max_cols if max_cols is not None else len(metric_cols)
        n_rows = len(metric_cols) // n_cols + (1 if len(metric_cols) % n_cols else 0)

    _, _, auto_figsize = get_layout(len(metric_cols), n_cols, figsize, strict_layout)
    figsize = figsize or auto_figsize
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    return fig, axs, group_to_alpha, alpha_to_group, base_colors, y_lim, n_rows, n_cols


def compute_pass_fail(
    group_metrics: List[Dict[str, Dict[str, float]]],
    group_list: List[str],
    metric: str,
    plot_thresholds: Tuple[float, float],
    raw_metrics: bool = False,
) -> Tuple[Dict[str, str], float, float]:
    """
    Compute pass/fail status and thresholds.

    raw_metrics: non disparity metrics; just raw performance
    group_metrics: list of dictionaries with group metrics
    """

    lower, upper = plot_thresholds
    if raw_metrics:
        # Disable pass/fail for raw metrics
        lower, upper = float("-inf"), float("inf")

    group_pass_fail = {}
    for row in group_metrics:
        for group in group_list:
            if group in row:  # Check if the group exists in the current row
                val = row[group][metric]  # Extract the metric value for the group
                # Append the value to the group's list in group_pass_fail
                # Use setdefault to initialize an empty list if the group isn't
                # in the dictionary yet
                group_pass_fail.setdefault(group, []).append(val)

    group_status = {
        group: "Pass" if all(lower <= v <= upper for v in vals) else "Fail"
        for group, vals in group_pass_fail.items()
    }
    return group_status, lower, upper


def create_legend(
    fig: plt.Figure,
    group_list: List[str],
    group_to_alpha: Dict[str, str],
    base_colors: Dict[str, str],
    show_pass_fail: bool,
    leg_cols: int = 6,
) -> None:
    """Create legends for group labels and pass/fail status."""
    group_legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="" if show_pass_fail else None,
            color=None if show_pass_fail else base_colors[group],
            lw=4,
            label=f"{group_to_alpha[group]}: {group}",
        )
        for group in group_list
    ]

    pass_fail_legend_handles = []
    if show_pass_fail:
        pass_fail_legend_handles = [
            Line2D([0], [0], color="green", lw=4, label="Pass"),
            Line2D([0], [0], color="red", lw=4, label="Fail"),
        ]

    fig.legend(
        handles=group_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=leg_cols,
        fontsize="large",
        frameon=False,
    )

    if show_pass_fail:
        fig.legend(
            handles=pass_fail_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(pass_fail_legend_handles),
            fontsize="large",
            frameon=False,
        )


################################################################################
# Residual Plot by Group
################################################################################


def _plot_residuals_ax(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    color: str,
    alpha: float = 0.6,
    show_centroid: bool = True,
    show_grid: bool = True,
) -> None:
    """Plot residuals for one group."""

    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=alpha, label=label, color=color)
    ax.axhline(0, **DEFAULT_LINE_KWARGS)
    if show_centroid:
        ax.scatter(
            np.mean(y_pred),
            np.mean(residuals),
            color=color,
            marker="X",
            s=120,
            edgecolor="black",
            linewidth=2,
            zorder=5,
        )
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title(str(label))
    ax.grid(show_grid)


def get_regression_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: str,
) -> str:
    """Generate label with regression metrics."""

    metrics = regression_metrics(y_true, y_pred)
    return (
        f"R² for {group} = {metrics['R^2 Score']:.2f}, "
        f"MAE = {metrics['Mean Absolute Error']:.2f}, "
        f"Residual μ = {metrics['Residual Mean']:.2f}, "
        f"n = {len(y_true):,}"
    )


def eq_plot_residuals_by_group(
    data: Dict[str, Dict[str, np.ndarray]],
    alpha: float = 0.6,
    show_centroids: bool = False,
    title: str = "Residuals by Group",
    filename: str = "residuals_by_group",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    subplots: bool = False,
    n_cols: int = 2,
    n_rows: Optional[int] = None,
    group: Optional[str] = None,
    color_by_group: bool = True,
    exclude_groups: Union[int, str, List[str], Set[str]] = 0,
    show_grid: bool = True,
) -> None:
    """Plot residuals grouped by subgroup."""

    # Check for NaN values in y_true and y_pred (or y_prob)
    _validate_plot_data(data, is_bootstrap=False)

    if group and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    def residual_plot(ax, data, group, color, overlay_mode=False):
        ax.clear() if not overlay_mode else None
        y_true = data[group].get("y_true", data[group].get("y_actual"))
        y_pred = data[group].get("y_prob", data[group].get("y_pred"))
        label = get_regression_label(y_true, y_pred, group)
        _plot_residuals_ax(
            ax,
            y_true,
            y_pred,
            label,
            color,
            alpha,
            show_centroids,
            show_grid=show_grid,
        )

    plot_with_layout(
        data,
        residual_plot,
        {},
        title=title,
        filename=filename,
        save_path=save_path,
        figsize=figsize,
        dpi=dpi,
        subplots=subplots,
        n_cols=n_cols,
        n_rows=n_rows,
        group=group,
        color_by_group=color_by_group,
        exclude_groups=exclude_groups,
        show_grid=show_grid,
    )


################################################################################
# Generic Group Curve Plotter
################################################################################


def _plot_group_curve_ax(
    ax: plt.Axes,
    data: Dict[str, Dict[str, np.ndarray]],
    group: str,
    color: str,
    curve_type: str = "roc",
    n_bins: int = 10,
    decimal_places: int = 2,
    label_mode: str = "full",
    curve_kwargs: Optional[Dict[str, Union[str, float]]] = None,
    line_kwargs: Optional[Dict[str, Union[str, float]]] = None,
    show_legend: bool = True,
    title: Optional[str] = None,
    is_subplot: bool = False,
    single_group: bool = False,
    show_grid: bool = True,
    lowess: float = 0,
    lowess_kwargs: Optional[Dict[str, Union[str, float]]] = None,
    shade_area: bool = False,
    hist: bool = (False,),
) -> None:
    y_true = data[group]["y_true"]
    y_prob = data[group]["y_prob"]
    total = len(y_true)
    positives = int(np.sum(y_true))
    negatives = total - positives

    curve_kwargs = curve_kwargs or {"color": color}
    line_kwargs = line_kwargs or DEFAULT_LINE_KWARGS
    lowess_kwargs = lowess_kwargs or {}

    if curve_type == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        score = auc(fpr, tpr)
        x, y = fpr, tpr
        x_label, y_label = "False Positive Rate", "True Positive Rate"
        ref_line = ([0, 1], [0, 1])
        prefix = "AUC"

        if label_mode == "simple":
            label = f"{prefix} = {score:.{decimal_places}f}"
        else:
            label = (
                f"{prefix} for {group} = {score:.{decimal_places}f}, "
                f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,}"
            )

    elif curve_type == "pr":
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        score = auc(recall, precision)
        x, y = recall, precision
        x_label, y_label = "Recall", "Precision"
        ref_line = ([0, 1], [positives / total] * 2)
        prefix = "AUCPR"

        if label_mode == "simple":
            label = f"{prefix} = {score:.{decimal_places}f}"
        else:
            label = (
                f"{prefix} for {group} = {score:.{decimal_places}f}, "
                f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,}"
            )

    elif curve_type == "calibration":
        # 1) get binned calibration
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        # 2) compute Brier for reference
        brier = brier_score_loss(y_true, y_prob)

        # compute calibration‐curve AUC via helper
        cal_auc = calibration_auc(mean_pred, frac_pos)

        # 4) assign plotting vars
        x, y = mean_pred, frac_pos
        x_label, y_label = "Mean Predicted Probability", "Fraction of Positives"
        ref_line = ([0, 1], [0, 1])

        # 5) custom label
        if label_mode == "simple":
            label = (
                f"Cal AUC = {cal_auc:.{decimal_places}f}, "
                f"Brier = {brier:.{decimal_places}f}, "
                f"Count = {total:,}"
            )

        else:
            label = (
                f"Cal AUC for {group} = {cal_auc:.{decimal_places}f}, "
                f"Brier = {brier:.{decimal_places}f}, "
                f"Count: {total:,}"
            )
        if shade_area:
            # 6) shade the area between the curve and the diagonal;
            #    first include the endpoints so the shading covers 0-1
            x_shade = np.concatenate(([0.0], x, [1.0]))
            y_shade = np.concatenate(([0.0], y, [1.0]))
            ax.fill_between(
                x_shade,
                y_shade,
                x_shade,  # the 45 degree line is y = x
                color=curve_kwargs.get("color", "gray"),
                alpha=0.2,
                label="_nolegend_",
            )
        # 7) draw a circle on each binned calibration point
        ax.scatter(
            x,
            y,
            marker="o",
            s=40,
            facecolor=curve_kwargs.get("color", color),
            zorder=5,
        )

        # --- draw inset histogram under the curve if requested ---
        # HISTOGRAM‐ONLY MODE: if requested, skip calibration curve
        if hist:
            ax.clear()
            # pick the exact same color the curve would have used
            hist_color = curve_kwargs.get("color", color)
            ax.hist(
                y_prob,
                bins=n_bins,
                range=(0, 1),
                color=hist_color,
                edgecolor="black",
            )
            ax.set_title(str(group))
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Count")
            ax.grid(False)
            return

    else:
        raise ValueError("Unsupported curve_type")

    #############  Common plotting
    ax.plot(x, y, label=label, **curve_kwargs)
    if lowess:
        # compute LOWESS smoothed curve
        smoothed = sm.nonparametric.lowess(y, x, frac=lowess)
        x_s, y_s = smoothed[:, 0], smoothed[:, 1]

        # reuse helpers to measure area
        lowess_auc = calibration_auc(x_s, y_s)

        # build the style for LOWESS: prefer lowess_kwargs → curve_kwargs → hard defaults
        smooth_kwargs = {
            "color": lowess_kwargs.get("color", curve_kwargs.get("color", "black")),
            "linestyle": lowess_kwargs.get(
                "linestyle", curve_kwargs.get("linestyle", ":")
            ),
            "linewidth": lowess_kwargs.get(
                "linewidth",
                lowess_kwargs.get("linewidth", curve_kwargs.get("linewidth", 1.5)),
            ),
        }
        # merge in any other valid plot keys the user passed to lowess_kwargs
        for k, v in lowess_kwargs.items():
            if k not in smooth_kwargs and k in VALID_PLOT_KWARGS:
                smooth_kwargs[k] = v

        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            label=f"LOWESS (AUC={lowess_auc:.3f})",
            **smooth_kwargs,
        )
    if curve_type != "pr":
        ax.plot(*ref_line, **line_kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_legend:
        # choose legend location per curve type & mode
        if is_subplot or single_group:
            loc = {
                "roc": "lower right",
                "pr": "upper right",
                "calibration": "lower right",
            }.get(curve_type, "best")
            legend_kwargs = {"loc": loc}
        else:
            legend_kwargs = DEFAULT_LEGEND_KWARGS
        ax.legend(**legend_kwargs)

    ax.grid(show_grid)
    ax.tick_params(axis="both")


def eq_plot_group_curves(
    data: Dict[str, Dict[str, np.ndarray]],
    curve_type: str = "roc",
    n_bins: int = 10,
    decimal_places: int = 2,
    curve_kwgs: Optional[Dict[str, Dict[str, Union[str, float]]]] = None,
    line_kwgs: Optional[Dict[str, Union[str, float]]] = None,
    title: str = "Curve by Group",
    filename: str = "group",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    subplots: bool = False,
    n_cols: int = 2,
    n_rows: Optional[int] = None,
    group: Optional[str] = None,
    color_by_group: bool = True,
    exclude_groups: Union[int, str, List[str], Set[str]] = 0,
    show_grid: bool = True,
    lowess: float = 0,
    lowess_kwargs: Optional[Dict[str, Union[str, float]]] = None,
    shade_area: bool = False,
    plot_hist: bool = False,
) -> None:
    """
    Plot ROC, PR, or calibration curves by group.

    data : dict - Mapping of group -> {'y_true': ..., 'y_prob': ...}
    curve_type : str - One of {'roc', 'pr', 'calibration'}
    n_bins : int - Number of bins for calibration curve
    decimal_places : int - Decimal precision for AUC or Brier score
    curve_kwgs : dict - Per-group matplotlib kwargs for curves
    line_kwgs : dict - Reference line style kwargs
    title : str - Plot title
    filename : str - Output filename (no extension)
    save_path : str or None - Directory to save plots if given
    figsize : tuple - Size of figure (w, h)
    dpi : int - Dots per inch (plot resolution)
    subplots : bool - Plot each group in a subplot
    n_cols : int - Number of subplot columns
    n_rows : int or None - Number of subplot rows
    group : str or None - If provided, plot only this group
    color_by_group : bool - Use different color per group
    exclude_groups : int|str|list|set - Exclude groups by name or sample size
    show_grid : bool - Toggle background grid on/off
    """

    # Validate plot data (check for missing y_true/y_prob and NaN values)
    _validate_plot_data(data, is_bootstrap=False)

    # Validate curve_kwgs and line_kwgs before proceeding
    _validate_plot_kwargs(curve_kwgs, data.keys(), kwarg_name="curve_kwgs")
    _validate_plot_kwargs(line_kwgs, valid_groups=None, kwarg_name="line_kwgs")

    if group and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")
    valid_data = _filter_groups(data, exclude_groups)

    if plot_hist:
        subplots = True

    def curve_plot(ax, data, group_iter, color, overlay_mode=False):
        # In overlay mode (subplots=False, group=None), use "full" label mode
        # In subplot or single-group mode, use "simple" label mode
        label_mode = "full" if overlay_mode else "simple"
        _plot_group_curve_ax(
            ax,
            data,
            group_iter,
            color,
            curve_type=curve_type,
            n_bins=n_bins,
            decimal_places=decimal_places,
            label_mode=label_mode,
            curve_kwargs=curve_kwgs.get(group_iter, {}) if curve_kwgs else None,
            line_kwargs=line_kwgs,
            show_legend=True,
            title=str(group_iter) if subplots else None,
            is_subplot=subplots,
            single_group=bool(group),
            show_grid=show_grid,
            lowess=lowess,
            lowess_kwargs=lowess_kwargs,
            shade_area=shade_area,
            hist=plot_hist,
        )

    plot_with_layout(
        valid_data,
        curve_plot,
        {},
        title=title,
        filename=filename,
        save_path=save_path,
        figsize=figsize,
        dpi=dpi,
        subplots=subplots,
        n_cols=n_cols,
        n_rows=n_rows,
        group=group,
        color_by_group=color_by_group,
        exclude_groups=exclude_groups,
        show_grid=show_grid,
    )


################################################################################
# Bootstrapped Group Curve Plot
################################################################################


def interpolate_bootstrapped_curves(
    boot_sliced_data: List[Dict[str, Dict[str, np.ndarray]]],
    grid_x: np.ndarray,
    curve_type: str = "roc",
    n_bins: int = 10,
) -> Tuple[Dict[str, List[np.ndarray]], np.ndarray]:
    """
    Interpolate bootstrapped curves over a common x-axis grid.

    boot_sliced_data : list of dict; each item represents a bootstrap iteration
                       with group-wise 'y_true' and 'y_prob' arrays.
    grid_x : np.ndarray; shared x-axis grid over which all curves will be interpolated.
    curve_type : str; type of curve to interpolate. One of {'roc', 'pr', 'calibration'}.
    n_bins : int; number of bins to use for calibration curves (ignored for 'roc' and 'pr').
    """

    result = {}
    if curve_type == "calibration":
        bins = np.linspace(0, 1, n_bins + 1)
        grid_x = (bins[:-1] + bins[1:]) / 2

    for bootstrap_iter in boot_sliced_data:
        for group, values in bootstrap_iter.items():
            y_true, y_prob = values["y_true"], values["y_prob"]
            try:
                if curve_type == "roc":
                    x_vals, y_vals, _ = roc_curve(y_true, y_prob)
                    # Interpolate TPR over the common FPR grid
                    interp_func = interp1d(
                        x_vals,
                        y_vals,
                        bounds_error=False,
                        fill_value=(0, 1),
                    )
                    y_interp = interp_func(grid_x)
                elif curve_type == "pr":
                    y_vals, x_vals, _ = precision_recall_curve(y_true, y_prob)
                    # Interpolate Precision over common Recall grid
                    interp_func = interp1d(
                        x_vals,
                        y_vals,
                        bounds_error=False,
                        fill_value=(0, 1),
                    )
                    y_interp = interp_func(grid_x)
                elif curve_type == "calibration":
                    # Manually compute average observed outcome per bin
                    y_interp = np.full(n_bins, np.nan)
                    for i in range(n_bins):
                        mask = (y_prob >= bins[i]) & (
                            (y_prob < bins[i + 1])
                            if i < n_bins - 1
                            else (y_prob <= bins[i + 1])
                        )
                        if np.any(mask):
                            y_interp[i] = np.mean(y_true[mask])
            except Exception:
                y_interp = np.full_like(
                    grid_x if curve_type != "calibration" else np.arange(n_bins),
                    np.nan,
                )
            result.setdefault(group, []).append(y_interp)
    return result, grid_x


def _plot_bootstrapped_curve_ax(
    ax: plt.Axes,
    y_array: np.ndarray,
    grid_x: np.ndarray,
    group: str,
    label_prefix: str = "AUROC",
    curve_kwargs: Optional[Dict[str, Union[str, float]]] = None,
    fill_kwargs: Optional[Dict[str, Union[str, float]]] = None,
    line_kwargs: Optional[Dict[str, Union[str, float]]] = None,
    show_grid: bool = True,
    bar_every: int = 10,
    brier_scores: Optional[Dict[str, List[float]]] = None,
    y_lim: Optional[Tuple[float, float]] = None,  # New parameter
) -> None:
    """Plot mean curve with confidence band and error bars for a bootstrapped group."""
    # Aggregate across bootstrap iterations
    mean_y = np.nanmean(y_array, axis=0)
    lower, upper = np.nanpercentile(y_array, [2.5, 97.5], axis=0)

    # non‐calibration AUCs (ROC/PR)
    aucs = (
        [np.trapz(y, grid_x) for y in y_array if not np.isnan(y).all()]
        if label_prefix != "CAL"
        else []
    )
    mean_auc = np.mean(aucs) if aucs else float("nan")
    low_auc, high_auc = (
        np.percentile(aucs, [2.5, 97.5]) if aucs else (float("nan"), float("nan"))
    )

    # Construct legend label depending on curve type

    if label_prefix == "CAL" and brier_scores:
        # … existing Brier logic …
        b_scores = brier_scores.get(group, [])
        mean_b = np.mean(b_scores) if b_scores else float("nan")
        low_b, high_b = (
            np.percentile(b_scores, [2.5, 97.5])
            if b_scores
            else (float("nan"), float("nan"))
        )

        # compute Cal-AUC *only* on fully populated bootstrap curves
        cal_aucs = []
        for y_row in y_array:
            if not np.isnan(y_row).any():  # drop rows with any NaNs
                cal_aucs.append(calibration_auc(grid_x, y_row))

        if cal_aucs:
            mean_c = np.mean(cal_aucs)
            low_c, high_c = np.percentile(cal_aucs, [2.5, 97.5])
        else:
            mean_c = low_c = high_c = float("nan")

        label = (
            f"{group}\n"
            f"(Mean Cal AUC = {mean_c:.3f} [{low_c:.3f},{high_c:.3f}];\n"
            f"Mean Brier = {mean_b:.3f} [{low_b:.3f},{high_b:.3f}])"
        )
    else:
        label = (
            f"{group} ({label_prefix} = {mean_auc:.2f} [{low_auc:.2f},{high_auc:.2f}])"
        )

    # Set default plotting styles
    curve_kwargs = curve_kwargs or {"color": "#1f77b4"}
    fill_kwargs = fill_kwargs or {
        "alpha": 0.2,
        "color": curve_kwargs.get("color", "#1f77b4"),
    }
    line_kwargs = line_kwargs or DEFAULT_LINE_KWARGS

    # Plot the average curve and its confidence band
    ax.plot(grid_x, mean_y, label=label, **curve_kwargs)
    ax.fill_between(grid_x, lower, upper, **fill_kwargs)

    # Add vertical error bars at regular intervals along the curve
    for j in range(0, len(grid_x), int(np.ceil(len(grid_x) / bar_every))):
        x_val, mean_val = grid_x[j], mean_y[j]
        ax.errorbar(
            x_val,
            mean_val,
            yerr=[[max(mean_val - lower[j], 0)], [max(upper[j] - mean_val, 0)]],
            fmt="o",
            color=curve_kwargs.get("color", "#1f77b4"),
            markersize=3,
            capsize=2,
            elinewidth=1,
            alpha=0.6,
        )

    # Add reference diagonal (for AUROC and CAL)
    if label_prefix in ["AUROC", "CAL"]:
        ax.plot([0, 1], [0, 1], **line_kwargs)
    ax.set_xlim(0, 1)

    # Set y-axis limits dynamically based on confidence intervals if not provided
    if y_lim is None:
        y_min = min(np.min(lower), 0.0)  # Ensure at least 0.0
        y_max = max(np.max(upper), 1.0)  # Ensure at least 1.0
        padding = 0.05 * (y_max - y_min)  # Add 5% padding
        y_lim = (y_min - padding, y_max + padding)
    ax.set_ylim(y_lim)

    ax.set_title(group)
    ax.set_xlabel(
        "False Positive Rate"
        if label_prefix == "AUROC"
        else "Recall" if label_prefix == "AUCPR" else "Mean Predicted Probability"
    )
    ax.set_ylabel(
        "True Positive Rate"
        if label_prefix == "AUROC"
        else "Precision" if label_prefix == "AUCPR" else "Fraction of Positives"
    )
    ax.grid(show_grid)
    ax.legend(loc="lower right")


def eq_plot_bootstrapped_group_curves(
    boot_sliced_data: List[Dict[str, Dict[str, np.ndarray]]],
    curve_type: str = "roc",
    common_grid: np.ndarray = np.linspace(0, 1, 100),
    bar_every: int = 10,
    n_bins: int = 10,
    line_kwgs: Optional[Dict[str, Union[str, float]]] = None,
    title: str = "Bootstrapped Curve by Group",
    filename: str = "bootstrapped_curve",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    subplots: bool = False,
    n_cols: int = 2,
    n_rows: Optional[int] = None,
    group: Optional[str] = None,
    color_by_group: bool = True,
    exclude_groups: Union[int, str, List[str], Set[str]] = 0,
    show_grid: bool = True,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot bootstrapped curves by group.

    boot_sliced_data : list - List of bootstrap iterations,
                       each as a dict of group -> {'y_true', 'y_prob'}
    curve_type : str - One of {'roc', 'pr', 'calibration'}
    common_grid : np.ndarray - Shared x-axis grid to interpolate curves over
    bar_every : int - Show error bars every N points along the curve
    n_bins : int - Number of bins used for calibration curves
    line_kwgs : dict - Reference line style kwargs
    title : str - Title for the plot
    filename : str - Output filename prefix (no extension)
    save_path : str or None - If given, save plot to this directory
    figsize : tuple - Size of the figure (width, height)
    dpi : int - Plot resolution (dots per inch)
    subplots : bool - Whether to plot each group in its own subplot
    n_cols : int - Number of subplot columns (ignored if subplots=False)
    n_rows : int or None - Number of subplot rows (auto if None)
    group : str or None - If provided, plot only for this group
    color_by_group : bool - Use different color per group
    exclude_groups : int|str|list|set - Exclude groups by name or sample size
    show_grid : bool - Whether to show gridlines on axes
    """

    # Validate plot data (check for missing y_true/y_prob and NaN values)
    _validate_plot_data(boot_sliced_data, is_bootstrap=True)

    # Validate line_kwgs before proceeding
    _validate_plot_kwargs(line_kwgs, valid_groups=None, kwarg_name="line_kwgs")

    interp_data, grid_x = interpolate_bootstrapped_curves(
        boot_sliced_data, common_grid, curve_type, n_bins
    )
    group_data = _get_concatenated_group_data(boot_sliced_data)
    valid_groups = _filter_groups(group_data, exclude_groups)
    interp_data = {g: interp_data[g] for g in valid_groups}

    label_prefix = (
        "AUROC" if curve_type == "roc" else "AUCPR" if curve_type == "pr" else "CAL"
    )
    brier_scores = (
        {
            g: [
                brier_score_loss(s[g]["y_true"], s[g]["y_prob"])
                for s in boot_sliced_data
                if g in s and len(set(s[g]["y_true"])) > 1
            ]
            for g in interp_data
        }
        if curve_type == "calibration"
        else None
    )

    def boot_plot(ax, interp_data, group, color, overlay_mode=False):
        ax.clear() if not overlay_mode else None
        valid_curves = [y for y in interp_data[group] if not np.isnan(y).all()]
        if not valid_curves:
            print(f"[Warning] Group '{group}' has no valid interpolated curves.")
            return
        y_array = np.vstack(valid_curves)
        _plot_bootstrapped_curve_ax(
            ax,
            y_array,
            grid_x,
            group,
            label_prefix,
            curve_kwargs={"color": color},
            brier_scores=brier_scores,
            bar_every=bar_every,
            show_grid=show_grid,
        )

    plot_with_layout(
        interp_data,
        boot_plot,
        {},
        title=title,
        filename=filename,
        save_path=save_path,
        figsize=figsize,
        dpi=dpi,
        subplots=subplots,
        n_cols=n_cols,
        n_rows=n_rows,
        group=group,
        color_by_group=color_by_group,
        exclude_groups=exclude_groups,
        show_grid=show_grid,
        y_lim=y_lim,
    )


################################################################################
# Group and Disparity Metrics (Violin/Box/Seaborn Plots)
################################################################################


def eq_group_metrics_plot(
    group_metrics: List[Dict[str, Dict[str, float]]],
    metric_cols: List[str],
    name: str,
    plot_type: str = "violinplot",
    categories: Union[str, List[str]] = "all",
    include_legend: bool = True,
    cmap: str = "tab20c",
    color_by_group: bool = True,
    save_path: Optional[str] = None,
    filename: str = "Disparity_Metrics",
    max_cols: Optional[int] = None,
    strict_layout: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    show_grid: bool = True,
    plot_thresholds: Tuple[float, float] = (0.0, 2.0),
    show_pass_fail: bool = False,
    leg_cols: int = 6,
    y_lim: Optional[Tuple[float, float]] = None,
    statistical_tests: dict = None,
    **plot_kwargs: Dict[str, Union[str, float]],
) -> None:
    """
    Plot group and disparity metrics as violin, box, or other seaborn plots with
    optional pass/fail coloring.

    group_metrics         : list           - One dict per category mapping group
    metric_cols           : list           - Metric names to plot
    name                  : str            - Plot title or identifier
    plot_kind             : str, default "violinplot" - Seaborn plot type
    categories            : str or list    - Categories to include or 'all'
    color_by_group        : bool, default True - Use separate colors per group
    max_cols              : int or None    - Max columns in facet grid
    strict_layout         : bool, default True - Apply tight layout adjustments
    plot_thresholds       : tuple, default (0.0, 2.0) - (lower, upper) for pass/fail
    show_pass_fail        : bool, default False - Color by pass/fail
    y_lim                 : tuple or None  - y-axis limits as (min, max)
    """

    if not isinstance(group_metrics, list):
        raise TypeError("group_metrics should be a list")

    all_keys = sorted({key for row in group_metrics for key in row.keys()})
    attributes = (
        [k for k in all_keys if k in categories] if categories != "all" else all_keys
    )

    # Shared setup
    n_cols = max_cols if max_cols is not None else len(metric_cols)
    fig, axs, group_to_alpha, alpha_to_group, base_colors, y_lim, n_rows, n_cols = (
        setup_plot_environment(
            attributes,
            cmap,
            color_by_group,
            metric_cols,
            max_cols,
            n_cols,
            figsize,
            strict_layout,
            y_lim,
            layout_type="violin",
        )
    )

    ## Initialise signficance checking
    significance_map = {}
    if statistical_tests:
        for group, metrics in statistical_tests.items():
            for metric_key, test_result in metrics.items():
                ## we have to remove _diff for it to work
                if metric_key in metric_cols and group in attributes:
                    significance_map[(group, metric_key)] = test_result.is_significant

    for i, col in enumerate(metric_cols):
        ax = axs[i // n_cols, i % n_cols]
        x_vals, y_vals = [], []

        if show_pass_fail:
            group_status, lower, upper = compute_pass_fail(
                group_metrics, attributes, col, plot_thresholds
            )

        for row in group_metrics:
            for attr in attributes:
                if attr in row:
                    val = row[attr][col]
                    x_vals.append(group_to_alpha[attr])
                    y_vals.append(val)

        if show_pass_fail:
            group_colors = {
                attr: "green" if group_status.get(attr) == "Pass" else "red"
                for attr in attributes
            }
        else:
            group_colors = base_colors

        plot_func = getattr(sns, plot_type, None)
        if not plot_func:
            raise ValueError(
                f"Unsupported plot_type: '{plot_type}'. Must be a valid seaborn plot type."
            )

        ## assemble common args
        plot_args = dict(
            ax=ax,
            x=x_vals,
            y=y_vals,
            hue=x_vals,
            palette={group_to_alpha[attr]: group_colors[attr] for attr in attributes},
        )
        ## only pass legend=False on newer seaborn
        if not SEABORN_OLD:
            plot_args["legend"] = False

        try:
            plot_func(**plot_args, **plot_kwargs)
        except TypeError as e:
            ## fallback for old seaborn which does not accept legend kwarg
            if SEABORN_OLD and "legend" in str(e):
                plot_args.pop("legend", None)
                plot_func(**plot_args, **plot_kwargs)
            else:
                raise ValueError(f"Failed to plot with {plot_type}: {e}")

        ## strip out the auto‐drawn legend on old seaborn
        if SEABORN_OLD:
            lg = ax.get_legend()
            if lg:
                lg.remove()

        ax.set_title(f"{name}_{col}")

        ax.set_xlabel("")
        ax.set_xticks(range(len(attributes)))
        labels = [
            group_to_alpha[attr]
            + (" *" if significance_map.get((attr, col), False) else "")
            for attr in attributes
        ]
        ax.set_xticklabels(labels, rotation=0, fontweight="bold")
        for tick_label in ax.get_xticklabels():
            # So our lookup doesn't break
            label_text = tick_label.get_text().replace(" *", "")
            attr = alpha_to_group[label_text]
            if show_pass_fail:
                tick_label.set_color(
                    "green" if group_status.get(attr) == "Pass" else "red"
                )
            else:
                tick_label.set_color(base_colors.get(attr, "black"))
        if show_pass_fail:
            add_plot_threshold_lines(ax, lower, upper, len(attributes))
        ax.set_ylim(y_lim)
        ax.grid(show_grid)

    for j in range(i + 1, n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")

    if include_legend:
        create_legend(
            fig, attributes, group_to_alpha, base_colors, show_pass_fail, leg_cols
        )

        if statistical_tests:
            stat_legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor="black",
                    markersize=10,
                    label="Statistically Significant Difference",
                ),
            ]
            stat_legend = fig.legend(
                handles=stat_legend_elements,
                loc="upper right",
                bbox_to_anchor=(0.7, 1.1),
            )

    if strict_layout:
        plt.tight_layout(w_pad=2, h_pad=2, rect=[0.01, 0.01, 1.01, 1])
    save_or_show_plot(fig, save_path, filename)


################################################################################
# Group and Disparity Metrics (Point Estimate Plots)
################################################################################


def eq_group_metrics_point_plot(
    group_metrics: List[Dict[str, Dict[str, float]]],
    metric_cols: List[str],
    category_names: List[str],
    include_legend: bool = True,
    cmap: str = "tab20c",
    save_path: Optional[str] = None,
    filename: str = "Point_Disparity_Metrics",
    strict_layout: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    show_grid: bool = True,
    plot_thresholds: Tuple[float, float] = (0.0, 2.0),
    show_pass_fail: bool = False,
    y_lim: Optional[Tuple[float, float]] = None,
    leg_cols: int = 3,
    raw_metrics: bool = False,
    statistical_tests: dict = None,
    show_reference: bool = True,
    y_lims: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = None,
    **plot_kwargs: Dict[str, Union[str, float]],
) -> None:
    """
    Plot point estimates of group and disparity metrics by category.

    group_metrics   : list of dict     - One dict per category mapping group
    metric_cols     : list             - Metric names to plot (defines rows)
    category_names  : list             - Category labels to plot (defines columns)
    cmap            : str              - Colormap for group coloring
    strict_layout   : bool             - Apply tight layout adjustments
    plot_thresholds : tuple            - (lower, upper) bounds for pass/fail
    show_pass_fail  : bool             - Color by pass/fail instead of group colors
    y_lim           : tuple or None    - y‑axis limits as (min, max)
    leg_cols        : int              - no. of columns in legend
    raw_metrics     : bool             - Treat metrics as raw; not metric ratios
    """
    # Determine all unique group names
    all_groups = sorted(set().union(*(gm.keys() for gm in group_metrics)))

    # Shared setup
    n_cols = len(category_names)
    fig, axs, group_to_alpha, alpha_to_group, base_colors, y_lim, _, n_cols = (
        setup_plot_environment(
            all_groups,
            cmap,
            True,
            metric_cols,
            None,
            n_cols,
            figsize,
            strict_layout,
            y_lim,
            layout_type="point",
        )
    )

    for i, metric in enumerate(metric_cols):
        for j, cat_name in enumerate(category_names):
            ax = axs[i, j]

            x_vals, y_vals = [], []
            groups = list(group_metrics[j].keys())
            # Create modified group labels for this category based on statistical tests
            current_group_to_alpha = group_to_alpha.copy()
            if statistical_tests and cat_name in statistical_tests:
                stat_tests = statistical_tests[cat_name]

                if stat_tests.get("omnibus") and stat_tests["omnibus"].is_significant:
                    current_group_to_alpha = {
                        grp: alph + " *" for grp, alph in current_group_to_alpha.items()
                    }

                for group in groups:
                    if group in stat_tests and stat_tests[group].is_significant:
                        current_group_to_alpha[group] += " ▲"

            current_alpha_to_group = {v: k for k, v in current_group_to_alpha.items()}

            group_status, lower, upper = compute_pass_fail(
                group_metrics, groups, metric, plot_thresholds, raw_metrics
            )

            for group in group_metrics[j]:
                val = group_metrics[j][group][metric]
                if not np.isnan(val):
                    x_vals.append(current_group_to_alpha[group])
                    y_vals.append(val)

            group_colors = (
                {
                    group: "green" if group_status.get(group) == "Pass" else "red"
                    for group in groups
                }
                if show_pass_fail
                else base_colors
            )

            for x, y, group in zip(range(len(x_vals)), y_vals, x_vals):
                sns.scatterplot(
                    x=[x],
                    y=[y],
                    ax=ax,
                    color=group_colors[current_alpha_to_group[group]],
                    s=100,
                    label=None,
                    **plot_kwargs,
                )

            ax.set_title(f"{cat_name}")
            ax.set_xlabel("")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(
                [current_group_to_alpha[group] for group in groups],
                rotation=45,
                ha="right",
            )

            for tick_label in ax.get_xticklabels():
                alpha = tick_label.get_text()
                group = current_alpha_to_group[alpha]
                if show_pass_fail:
                    color = "green" if group_status[group] == "Pass" else "red"
                else:
                    color = base_colors[group]  # legend color
                tick_label.set_color(color)

            if j == 0:
                ax.set_ylabel(metric)
            else:
                ax.set_ylabel("")

            if y_lims and (i, j) in y_lims:
                ax.set_ylim(y_lims[(i, j)])
            else:
                ax.set_ylim(y_lim)
            ax.grid(show_grid)
            add_plot_threshold_lines(ax, lower, upper, len(groups), show_reference)
            ax.set_xlim(-0.5, len(groups) - 0.5)

    for row_idx in range(len(metric_cols)):
        for col_idx in range(len(category_names), n_cols):
            if col_idx < n_cols:
                axs[row_idx, col_idx].axis("off")

    if include_legend:
        create_legend(
            fig, all_groups, group_to_alpha, base_colors, show_pass_fail, leg_cols
        )

        if statistical_tests:

            stat_legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor="black",
                    markersize=10,
                    label="Omnibus test significant",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="black",
                    markersize=8,
                    label="Group test significant",
                ),
            ]
            stat_legend = fig.legend(
                handles=stat_legend_elements,
                loc="upper right",
                bbox_to_anchor=(0.7, 1.1),
            )

    if strict_layout:
        plt.tight_layout(w_pad=2, h_pad=4, rect=[0.01, 0.01, 1.01, 1])
    save_or_show_plot(fig, save_path, filename)


def eq_plot_metrics_forest(
    group_metrics: Dict[str, Dict[str, float]],
    metric_name: str,
    reference_group: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
    save_path: Optional[str] = None,
    filename: str = "points_forest",
    sort_groups: bool = True,
    sort_alphabetically: bool = False,
    ascending: bool = True,
    title: str = None,
    statistical_tests: Optional[Dict[str, bool]] = None,
    x_lim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Create a forest plot of point estimates for a specific metric across groups.
    """
    valid_groups = {
        group: metrics
        for group, metrics in group_metrics.items()
        if metric_name in metrics and not np.isnan(metrics[metric_name])
    }

    if not valid_groups:
        raise ValueError(f"No valid data found for metric '{metric_name}'")

    if reference_group and reference_group not in valid_groups:
        raise ValueError(
            f"Reference group '{reference_group}' not found in valid groups: {list(valid_groups.keys())}"
        )

    groups = list(valid_groups.keys())
    values = [valid_groups[group][metric_name] for group in groups]

    # Add statistical significance markers
    if statistical_tests:
        for group in groups:
            if group in statistical_tests and statistical_tests[group].is_significant:
                groups[groups.index(group)] += " ▲"

        if (
            statistical_tests.get("omnibus")
            and statistical_tests["omnibus"].is_significant
        ):
            groups = [f"{group} *" for group in groups]

    if sort_groups:
        sorted_pairs = sorted(
            zip(groups, values), key=lambda x: x[1], reverse=not ascending
        )
        groups, values = zip(*sorted_pairs)
        groups, values = list(groups), list(values)

    # Optional alphabetical sorting
    if sort_alphabetically:
        sorted_pairs = sorted(zip(groups, values), key=lambda x: x[0])
        groups, values = zip(*sorted_pairs)
        groups, values = list(groups), list(values)

    y_pos = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(values, y_pos, s=64, color="black", zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric_name}")

    if title is None:
        ax.set_title(f"Forest Plot: {metric_name} by Group")
    else:
        ax.set_title(title)

    ax.grid(True, alpha=0.3, zorder=1)

    if reference_group:
        ref_value = valid_groups[reference_group][metric_name]
        ax.axvline(
            x=ref_value,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Ref. group",
        )

    if x_lim is not None:
        ax.set_xlim(x_lim)

    if statistical_tests:
        legend_elements = []

        legend_elements.append(
            Line2D(
                [0],
                [0],
                linestyle="--",
                color="red",
                markerfacecolor="black",
                markersize=8,
                label="Ref. group",
            )
        )

        if (
            statistical_tests.get("omnibus")
            and statistical_tests["omnibus"].is_significant
        ):
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor="black",
                    markersize=10,
                    label="Omnibus test significant",
                    linestyle="None",
                )
            )

        original_groups = list(valid_groups.keys())
        group_tests_significant = any(
            group in statistical_tests and statistical_tests[group].is_significant
            for group in original_groups
        )

        if group_tests_significant:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="black",
                    markersize=8,
                    label="Group test significant",
                    linestyle="None",
                )
            )

        # Add legend if there are elements to show
        if legend_elements:
            ax.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    save_or_show_plot(fig, save_path, filename)


def eq_plot_bootstrap_forest(
    group_boot_metrics: List[Dict[str, Dict[str, np.ndarray]]],
    metric: str = "Accuracy",
    reference_group: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
    save_path: Optional[str] = None,
    filename: str = "bootstrap_forest",
    title: str = None,
    statistical_tests: Optional[Dict[str, Dict[str, bool]]] = None,
    x_lim: Optional[Tuple[float, float]] = None,
    sort_alphabetically: bool = False,
) -> None:
    """
    Create a forest plot of any bootstrap metric with 95% CI for each
    group, and draw a dotted line through `reference_group` if provided.
    Adds asterisks to group names with significant differences.
    """
    # Calculate mean and 95% CI for each metric per group
    bootstrap_metrics = calculate_bootstrap_stats(group_boot_metrics, metric)
    fig, ax = plt.subplots(figsize=figsize)

    # Sort groups for consistent ordering (optional)
    # Sorting logic
    if sort_alphabetically:
        stats_df = bootstrap_metrics.sort_values("group", ascending=True)
    else:
        stats_df = bootstrap_metrics.sort_values("mean", ascending=True)

    y_pos = np.arange(len(stats_df))

    # Prepare group labels with significance indicators
    group_labels = []
    has_significant = False
    if statistical_tests:
        metric_diff_key = f"{metric}_diff"
        for group in stats_df["group"]:
            if (
                group in statistical_tests
                and metric_diff_key in statistical_tests[group]
                and statistical_tests[group][metric_diff_key].is_significant
            ):
                group_labels.append(f"{group}  $\\mathbf{{*}}$")
                has_significant = True
            else:
                group_labels.append(group)
    else:
        group_labels = stats_df["group"].tolist()

    for i, (_, row) in enumerate(stats_df.iterrows()):
        ax.plot(
            [row["ci_lower"], row["ci_upper"]],
            [i, i],
            "k-",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            [row["ci_lower"], row["ci_lower"]],
            [i - 0.1, i + 0.1],
            "k-",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            [row["ci_upper"], row["ci_upper"]],
            [i - 0.1, i + 0.1],
            "k-",
            linewidth=2,
            alpha=0.7,
        )

    # Plot means as dots
    colors = ["black" for group in stats_df["group"]]
    ax.scatter(stats_df["mean"], y_pos, color=colors, s=75, zorder=5, alpha=0.8)

    # Add reference line if specified
    if reference_group and reference_group in stats_df["group"].values:
        ref_mean = stats_df[stats_df["group"] == reference_group]["mean"].iloc[0]
        ax.axvline(
            x=ref_mean,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Ref. group",
        )

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_labels)  # Use labels with asterisks
    ax.set_xlabel(f"{metric} (95% CI)")
    ax.set_ylabel("Groups")

    if title is None:
        title = f"Forest Plot: {metric} by Group"
    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3, axis="x")

    # Add legends
    legend_elements = []
    if reference_group:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Ref. group",
            )
        )
    if has_significant:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="black",
                markersize=10,
                label="Omnibus test significant",
                linestyle="None",
            )
        )

    if legend_elements:
        ax.legend(handles=legend_elements)

    if x_lim is not None:
        ax.set_xlim(x_lim)

    plt.tight_layout()
    save_or_show_plot(fig, save_path, filename)


################################################################################
############################ Effect Size Plots #################################

## EquiBoots also calculates effect size when we are dealing with point
# estimates. In this case we can see the effect size for all of the results is
# low (under 0.2) with the highest being 0.11. This indicates that although
# statistical signficance was found it is not necessary a strong finding.
################################################################################


def plot_effect_sizes(
    stat_test_results,
    xlabel="Category",
    ylabel="Effect size",
    title="Effect Sizes by Group",
    figsize=(8, 6),
    rotation=0,
    legend_loc="best",
    save_path=None,
    filename="effect_sizes",
):
    """
    Bar chart of effect sizes by group.
    Expects a mapping of labels to objects with .effect_size. Bars are annotated,
    horizontal guides mark 0.2 and 0.6, and a PNG is saved if `save_path` is provided.
    """
    stat_results_keys = stat_test_results.keys()
    effect_sizes = [
        result.effect_size if result.effect_size else 0
        for _, result in stat_test_results.items()
    ]

    plt.figure(figsize=figsize)
    # Create the bar chart
    bars = plt.bar(stat_results_keys, effect_sizes)

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            round(yval, 2),
            ha="center",
            va="bottom",
        )  # Adjust vertical position (0.01) as needed

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Add horizontal lines for effect size ranges
    plt.axhline(y=0.2, color="red", linestyle="--", label="Small effect size <= 0.2")
    plt.axhline(y=0.6, color="red", linestyle="--", label="Medium effect size <= 0.6")
    plt.plot([], [], color="red", linestyle="--", label="Large effect size > 0.6")

    # Add a legend and grid
    plt.legend(loc=legend_loc)
    plt.grid(axis="y", linestyle="--")

    # decide horizontal alignment based on rotation
    ha = "center" if rotation == 0 else "right"
    plt.xticks(rotation=rotation, ha=ha)
    plt.tight_layout()

    # save or show
    fig = plt.gcf()
    if save_path:
        fig.savefig(os.path.join(save_path, f"{filename}.png"), bbox_inches="tight")
    plt.show()
