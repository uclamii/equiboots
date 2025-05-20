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
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import seaborn as sns

from .metrics import regression_metrics, calibration_auc
from typing import Dict, List, Optional, Union, Tuple, Set, Callable

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

    if show_grid:
        plt.grid(linestyle=":")

    save_or_show_plot(fig, save_path, filename)


def add_plot_threshold_lines(
    ax: plt.Axes, lower: float, upper: float, xmax: float
) -> None:
    """Add disparity threshold lines to the plot."""
    ax.hlines(
        [lower, 1.0, upper],
        xmin=-0.5,
        xmax=xmax + 0.5,
        ls=":",
        colors=["red", "black", "red"],
    )
    ax.set_xlim(-0.5, xmax + 0.5)


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
) -> None:
    y_true = data[group]["y_true"]
    y_prob = data[group]["y_prob"]
    total = len(y_true)
    positives = int(np.sum(y_true))
    negatives = total - positives

    curve_kwargs = curve_kwargs or {"color": color}
    line_kwargs = line_kwargs or DEFAULT_LINE_KWARGS

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
        x_label, y_label = "Mean Predicted Value", "Fraction of Positives"
        ref_line = ([0, 1], [0, 1])

        # 5) custom label
        if label_mode == "simple":
            label = f"Cal AUC = {cal_auc:.{decimal_places}f}"
        else:
            label = (
                f"Cal AUC for {group} = {cal_auc:.{decimal_places}f}, "
                f"Brier = {brier:.{decimal_places}f}, "
                f"Count: {total:,}"
            )

    else:
        raise ValueError("Unsupported curve_type")

    #############  Common plotting
    ax.plot(x, y, label=label, **curve_kwargs)
    if curve_type == "calibration":
        ax.scatter(x, y, color=curve_kwargs.get("color", "black"), zorder=5)
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

    # Calculate AUC summary stats if not calibration
    aucs = (
        [np.trapz(y, grid_x) for y in y_array if not np.isnan(y).all()]
        if label_prefix != "CAL"
        else []
    )
    mean_auc = np.mean(aucs) if aucs else float("nan")
    lower_auc, upper_auc = (
        np.percentile(aucs, [2.5, 97.5]) if aucs else (float("nan"), float("nan"))
    )

    # Construct legend label depending on curve type
    if label_prefix == "CAL" and brier_scores:
        scores = brier_scores.get(group, [])
        mean_brier = np.mean(scores) if scores else float("nan")
        lower_brier, upper_brier = (
            np.percentile(scores, [2.5, 97.5])
            if scores
            else (float("nan"), float("nan"))
        )
        label = f"{group} (Mean Brier = {mean_brier:.3f} [{lower_brier:.3f}, {upper_brier:.3f}])"
    else:
        label = f"{group} ({label_prefix} = {mean_auc:.2f} [{lower_auc:.2f}, {upper_auc:.2f}])"

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
    )


################################################################################
# Group and Disparity Metrics (Violin/Box/Seaborn Plots)
################################################################################


def eq_group_metrics_plot(
    group_metrics: List[Dict[str, Dict[str, float]]],
    metric_cols: List[str],
    name: str,
    plot_kind: str = "violinplot",
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
    y_lim: Optional[Tuple[float, float]] = None,
    **plot_kwargs: Dict[str, Union[str, float]],
) -> None:
    """
    Plot group and disparity metrics as violin, box, or other seaborn plots with
    optional pass/fail coloring.

    group_metrics         : list           - One dict per category mapping group
                                             -> {metric: value}
    metric_cols           : list           - Metric names to plot
    name                  : str            - Plot title or identifier
    plot_kind             : str, default "violinplot" - Seaborn plot type (e.g.,
                                                      'violinplot', 'boxplot')
    categories            : str or list    - Categories to include or 'all'
    color_by_group        : bool, default True - Use separate colors per group
    max_cols              : int or None    - Max columns in facet grid
    strict_layout         : bool, default True - Apply tight layout adjustments
    plot_thresholds  : tuple, default (0.0, 2.0) - (lower, upper) bounds for
                                                        pass/fail
    show_pass_fail        : bool, default False - Color by pass/fail instead of
                                                  group colors
    y_lim                 : tuple or None  - y-axis limits as (min, max)
    """

    if not isinstance(group_metrics, list):
        raise TypeError("group_metrics should be a list")

    all_keys = sorted({key for row in group_metrics for key in row.keys()})
    attributes = (
        [k for k in all_keys if k in categories] if categories != "all" else all_keys
    )

    color_map = plt.get_cmap(cmap)
    colors = [color_map(i / len(attributes)) for i in range(len(attributes))]
    base_colors = {
        attr: (colors[i] if color_by_group else "#1f77b4")
        for i, attr in enumerate(attributes)
    }
    legend_colors = {attr: colors[i] for i, attr in enumerate(attributes)}

    n_rows, n_cols, auto_figsize = get_layout(
        len(metric_cols), max_cols, figsize, strict_layout
    )
    if figsize is None:
        figsize = auto_figsize
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    if y_lim is None:  # Set default y_lim if not specified
        y_lim = (-2, 4)  # Always default to (-2, 4) if not specified

    for i, col in enumerate(metric_cols):
        ax = axs[i // n_cols, i % n_cols]
        x_vals, y_vals = [], []
        group_pass_fail = {}

        for row in group_metrics:
            for attr in attributes:
                if attr in row:
                    val = row[attr][col]
                    x_vals.append(attr)
                    y_vals.append(val)
                    group_pass_fail.setdefault(attr, []).append(val)

        lower, upper = plot_thresholds

        group_status = {
            attr: "Pass" if all(lower <= v <= upper for v in vals) else "Fail"
            for attr, vals in group_pass_fail.items()
        }

        group_colors = (
            {
                attr: ("green" if group_status.get(attr) == "Pass" else "red")
                for attr in attributes
            }
            if show_pass_fail
            else base_colors
        )

        plot_func = getattr(sns, plot_kind, None)
        if not plot_func:
            raise ValueError(
                f"Unsupported plot_kind: '{plot_kind}'. Must be a seaborn plot type."
            )
        plot_func(
            ax=ax,
            x=x_vals,
            y=y_vals,
            hue=x_vals,
            palette=group_colors,
            legend=False,
            **plot_kwargs,
        )

        ax.set_title(f"{name}_{col}")

        ax.set_xlabel("")
        ax.set_xticks(range(len(attributes)))
        ax.set_xticklabels(attributes, rotation=0, fontweight="bold")
        for tick_label in ax.get_xticklabels():
            attr = tick_label.get_text()
            tick_label.set_color(
                ("green" if group_status.get(attr) == "Pass" else "red")
                if show_pass_fail
                else legend_colors.get(attr, "black")
            )
        add_plot_threshold_lines(ax, lower, upper, len(attributes))
        ax.set_ylim(y_lim)
        ax.grid(show_grid)

    for j in range(i + 1, n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")

    if include_legend:
        if show_pass_fail:
            legend_handles = [
                Line2D([0], [0], color="green", lw=4, label="Pass"),
                Line2D([0], [0], color="red", lw=4, label="Fail"),
            ]
        else:
            legend_handles = [
                Line2D([0], [0], color=legend_colors[attr], lw=4, label=attr)
                for attr in attributes
            ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(legend_handles),
            fontsize="large",
            frameon=False,
        )

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
    raw_metrics: bool = False,
    **plot_kwargs: Dict[str, Union[str, float]],
) -> None:
    """
    Plot point estimates of group and disparity metrics by category.

    group_metrics   : list of dict     - One dict per category mapping group ->
                                         {metric: value}
    metric_cols     : list             - Metric names to plot (defines rows)
    category_names  : list             - Category labels to plot (defines columns)
    cmap            : str              - Colormap for group coloring
    save_path       : str or None      - Directory to save figure (None displays)
    filename        : str              - Filename prefix (no extension)
    strict_layout   : bool             - Apply tight layout adjustments
    plot_thresholds : tuple            - (lower, upper) bounds for pass/fail
    show_pass_fail  : bool             - Color by pass/fail instead of group colors
    y_lim           : tuple or None    - y‑axis limits as (min, max)
    raw_metrics     : bool             - Treat metrics as raw; not metric ratios
    """

    # Set up colors
    color_map = plt.get_cmap(cmap)
    all_groups = sorted({group for groups in group_metrics for group in groups})
    colors = [color_map(i / len(all_groups)) for i in range(len(all_groups))]
    base_colors = {group: colors[i] for i, group in enumerate(all_groups)}

    # Compute layout: rows = metrics, columns = categories
    n_rows = len(metric_cols)  # One row per metric
    n_cols = len(category_names)  # One column per category

    # Use the existing get_layout function to determine figure size
    _, _, auto_figsize = get_layout(
        n_items=n_cols, n_cols=n_cols, figsize=figsize, strict_layout=strict_layout
    )
    figsize = figsize or auto_figsize

    # Create subplot grid: rows = metrics, columns = categories
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    lower, upper = plot_thresholds

    if raw_metrics:
        # Raw numbers --> skip pass/fail colouring + disable thresholds.
        show_pass_fail = False
        # Force infinite thresholds so the “Fail” check can never fire
        lower, upper = float("-inf"), float("inf")

    for i, metric in enumerate(metric_cols):
        for j, cat_name in enumerate(category_names):
            ax = axs[i, j]  # Access subplot: row = metric, column = category

            x_vals = []
            y_vals = []
            group_pass_fail = {}
            groups = list(group_metrics[j].keys())
            for group in group_metrics[j]:
                val = group_metrics[j][group][metric]
                if not np.isnan(val):
                    x_vals.append(group)
                    y_vals.append(val)
                    group_pass_fail.setdefault(group, []).append(val)

            # Determine pass/fail status for each group
            group_status = {
                group: "Pass" if all(lower <= v <= upper for v in vals) else "Fail"
                for group, vals in group_pass_fail.items()
            }

            group_colors = (
                {
                    group: ("green" if group_status.get(group) == "Pass" else "red")
                    for group in groups
                }
                if show_pass_fail
                else base_colors
            )

            # Plot points
            for x, y, group in zip(range(len(x_vals)), y_vals, x_vals):
                sns.scatterplot(
                    x=[x],
                    y=[y],
                    ax=ax,
                    color=group_colors[group],
                    s=100,
                    label=None,  # No subplot legend
                    **plot_kwargs,
                )

            # Customize axis
            ax.set_title(f"{cat_name} - {metric}")
            ax.set_xlabel("")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=45, ha="right")
            for tick_label in ax.get_xticklabels():
                group = tick_label.get_text()
                if show_pass_fail:
                    tick_label.set_color(
                        "green" if group_status.get(group) == "Pass" else "red"
                    )
                else:
                    tick_label.set_color(base_colors.get(group, "black"))

            ax.set_ylim(y_lim)
            ax.grid(show_grid)

            add_plot_threshold_lines(ax, lower, upper, len(groups))
            ax.set_xlim(-0.5, len(groups) - 0.5)

    # Turn off unused subplots (shouldn't be needed since grid matches exactly)
    for row_idx in range(len(metric_cols)):
        for col_idx in range(len(category_names), n_cols):
            if col_idx < n_cols:  # Just in case
                axs[row_idx, col_idx].axis("off")

    # Add overarching legend
    if include_legend:
        if show_pass_fail:
            legend_handles = [
                Line2D([0], [0], color="green", lw=4, label="Pass"),
                Line2D([0], [0], color="red", lw=4, label="Fail"),
            ]
        else:
            legend_handles = [
                Line2D([0], [0], color=base_colors[group], lw=4, label=group)
                for group in all_groups
            ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(legend_handles),
            fontsize="large",
            frameon=False,
        )

    if strict_layout:
        plt.tight_layout(w_pad=2, h_pad=2, rect=[0.01, 0.01, 1.01, 1])

    save_or_show_plot(fig, save_path, filename)
