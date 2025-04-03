# Consolidated Group Plotting Utilities
from scipy.interpolate import interp1d
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
from matplotlib.lines import Line2D
import seaborn as sns

from .metrics import regression_metrics

################################################################################
# Shared Utilities
################################################################################


def save_or_show_plot(fig, save_path=None, filename="plot", bbox_inches="tight"):
    """Save plot to file if path is provided, otherwise display it."""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"{filename}.png"), bbox_inches=bbox_inches)
    plt.show()


def get_group_color_map(groups, palette="tab10"):
    """Generate a mapping from group names to colors."""
    colors = plt.get_cmap(palette).colors
    return {g: colors[i % len(colors)] for i, g in enumerate(groups)}


def default_line_kwargs():
    """Return default line style kwargs for reference lines."""
    return {"color": "gray", "linestyle": "--", "linewidth": 1}


def _validate_groups(data):
    """Filter out groups with only one class in y_true."""
    return {g: v for g, v in data.items() if len(set(v["y_true"])) > 1}


def get_layout(n_items, n_cols=None, figsize=None, strict_layout=True):
    """Compute layout grid and figure size based on number of items."""
    if strict_layout:
        if n_cols is None:
            n_cols = 6
        n_rows = int(np.ceil(n_items / n_cols))
        fig_width = 24 if figsize is None else figsize[0]
        fig_height = 4 * n_rows if figsize is None else figsize[1]
    else:
        if figsize is not None:
            if n_cols is None:
                n_cols = int(np.ceil(np.sqrt(n_items)))
            n_rows = int(np.ceil(n_items / n_cols))
            fig_width, fig_height = figsize
        else:
            if n_cols is None:
                n_cols = int(np.ceil(np.sqrt(n_items)))
            n_rows = int(np.ceil(n_items / n_cols))
            fig_width = 5 * n_cols
            fig_height = 5 * n_rows

    return n_rows, n_cols, (fig_width, fig_height)


################################################################################
# Residual Plot by Group
################################################################################


def _plot_residuals_ax(
    ax,
    y_true,
    y_pred,
    label,
    color,
    alpha=0.6,
    show_centroid=True,
):
    """Plot residuals of predictions vs true values for one group."""
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=alpha, label=label, color=color)
    ax.axhline(0, **default_line_kwargs())
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
    ax.grid(True)


def get_regression_label(y_true, y_pred, group):
    metrics = regression_metrics(y_true, y_pred)
    label = (
        f"R² for {group} = {metrics['R^2 Score']:.2f}, "
        f"MAE = {metrics['Mean Absolute Error']:.2f}, "
        f"Residual μ = {metrics['Residual Mean']:.2f}, "
        f"n = {len(y_true):,}"
    )
    return label


def eq_plot_residuals_by_group(
    data,
    save_path=None,
    filename="residuals_by_group",
    title="Residuals by Group",
    figsize=(8, 6),
    dpi=100,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    color_by_group=True,
    alpha=0.6,
    show_centroids=False,
):
    """
    Plot residuals grouped by subgroup.

    - data: dict of group -> {y_true, y_pred/y_prob}
    - save_path: optional output directory to save figure
    - filename: base filename for saving
    - title: plot or grid title
    - figsize: tuple for figure size per plot
    - dpi: plot resolution
    - subplots: display per-group subplots if True
    - n_cols: number of columns for subplot grid
    - n_rows: optional manual override for subplot rows
    - group: optionally plot only this group
    - color_by_group: color points by group if True, uniform if False
    - alpha: transparency of scatter points
    - show_centroids: whether to mark group centroids with X
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    groups = sorted(data.keys())
    color_map = (
        get_group_color_map(groups)
        if color_by_group
        else {g: "#1f77b4" for g in groups}
    )

    if group:
        if group not in data:
            print(f"[Warning] Group '{group}' not found.")
            return
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        d = data[group]
        y_true = d.get("y_true", d.get("y_actual"))
        y_pred = d.get("y_prob", d.get("y_pred"))
        label = get_regression_label(y_true, y_pred, group)
        _plot_residuals_ax(
            ax, y_true, y_pred, label, color_map[group], alpha, show_centroids
        )
        ax.set_title(f"{title} ({group})")
        fig.tight_layout()
        save_or_show_plot(fig, save_path, f"{filename}_{group}")
        return

    if subplots:
        if n_rows is None:
            n_rows = int(np.ceil(len(groups) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), dpi=dpi
        )
        axes = axes.flatten()
        for i, g in enumerate(groups):
            if i >= len(axes):
                break
            d = data[g]
            y_true = d.get("y_true", d.get("y_actual"))
            y_pred = d.get("y_prob", d.get("y_pred"))
            label = get_regression_label(y_true, y_pred, g)
            _plot_residuals_ax(
                axes[i], y_true, y_pred, label, color_map[g], alpha, show_centroids
            )
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_or_show_plot(fig, save_path, filename)
        return

    # Combined overlay
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for g in groups:
        d = data[g]
        y_true = d.get("y_true", d.get("y_actual"))
        y_pred = d.get("y_prob", d.get("y_pred"))
        label = get_regression_label(y_true, y_pred, g)
        _plot_residuals_ax(
            ax, y_true, y_pred, label, color_map[g], alpha, show_centroids
        )
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show_plot(fig, save_path, f"{filename}_overlay")


################################################################################
# Generic Group Curve Plotter
################################################################################


def _plot_group_curve_ax(
    ax,
    data,
    group,
    curve_type="roc",
    n_bins=10,
    decimal_places=2,
    label_mode="full",
    curve_kwargs=None,
    line_kwargs=None,
    show_legend=True,
    title=None,
):
    """Plot ROC, PR, or calibration curve for a single group."""

    y_true = data[group]["y_true"]
    y_prob = data[group]["y_prob"]
    total = len(y_true)
    positives = int(np.sum(y_true))
    negatives = total - positives

    curve_kwargs = curve_kwargs or {}
    line_kwargs = line_kwargs or default_line_kwargs()

    if curve_type == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        score = auc(fpr, tpr)
        x, y = fpr, tpr
        x_label, y_label = "False Positive Rate", "True Positive Rate"
        ref_line = ([0, 1], [0, 1])
        prefix = "AUC"
    elif curve_type == "pr":
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        y, x = precision, recall
        score = auc(recall, precision)
        x_label, y_label = "Recall", "Precision"
        ref_line = ([0, 1], [positives / total] * 2)
        prefix = "AUCPR"
    elif curve_type == "calibration":
        y, x = calibration_curve(y_true, y_prob, n_bins=n_bins)
        score = brier_score_loss(y_true, y_prob)
        x_label, y_label = "Mean Predicted Value", "Fraction of Positives"
        ref_line = ([0, 1], [0, 1])
        prefix = "Brier"
    else:
        raise ValueError("Unsupported curve_type")

    if label_mode == "simple":
        label = f"{prefix} = {score:.{decimal_places}f}"
    else:
        label = (
            f"{prefix} for {group} = {score:.{decimal_places}f}, "
            f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,}"
        )

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
        ax.legend()
    ax.grid(True)
    ax.tick_params(axis="both")


################################################################################
# Unified Group Plot Wrapper
################################################################################


def eq_plot_group_curves(
    data,
    curve_type="roc",
    save_path=None,
    filename="group_curve",
    title="Curve by Group",
    figsize=(8, 6),
    dpi=100,
    decimal_places=2,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    color_by_group=True,
    curve_kwgs=None,
    line_kwgs=None,
    n_bins=10,
):
    """
    Plot ROC, PR, or calibration curves by group with optional subplots.

    - data: dict of group -> {y_true, y_prob}
    - curve_type: 'roc', 'pr', or 'calibration'
    - save_path: directory to save plots, if provided
    - filename: base filename for saved plot
    - title: title of the plot or grid
    - figsize: size of figure per subplot or combined plot
    - dpi: resolution of the figure
    - decimal_places: precision for displayed metrics
    - subplots: display group curves in a grid layout
    - n_cols: number of subplot columns
    - n_rows: number of subplot rows (auto-calculated if None)
    - group: if set, only plot this specific group
    - color_by_group: if True, use different colors per group
    - curve_kwgs: optional per-group styling for curves
    - line_kwgs: optional styling for reference line
    - n_bins: number of bins for calibration curves
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    valid_data = _validate_groups(data)
    groups = sorted(valid_data.keys())

    color_map = (
        get_group_color_map(groups)
        if color_by_group
        else {g: "#1f77b4" for g in groups}
    )

    final_curve_kwgs = {
        g: {
            "color": color_map.get(g, "gray"),
            **(curve_kwgs.get(g, {}) if curve_kwgs else {}),
        }
        for g in groups
    }

    if group:
        if group not in valid_data:
            print(f"[Warning] Group '{group}' not found or invalid.")
            return
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot_group_curve_ax(
            ax=ax,
            data=valid_data,
            group=group,
            curve_type=curve_type,
            n_bins=n_bins,
            decimal_places=decimal_places,
            label_mode="simple",
            curve_kwargs=final_curve_kwgs[group],
            line_kwargs=line_kwgs,
            title=f"{title} ({group})",
        )
        fig.tight_layout()
        save_or_show_plot(fig, save_path, f"{filename}_{group}")
        return

    if subplots:
        if n_rows is None:
            n_rows = int(np.ceil(len(groups) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), dpi=dpi
        )
        axes = axes.flatten()
        for i, g in enumerate(groups):
            if i >= len(axes):
                break
            _plot_group_curve_ax(
                ax=axes[i],
                data=valid_data,
                group=g,
                curve_type=curve_type,
                n_bins=n_bins,
                decimal_places=decimal_places,
                label_mode="simple",
                curve_kwargs=final_curve_kwgs[g],
                line_kwargs=line_kwgs,
                title=str(g),
            )
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_or_show_plot(fig, save_path, filename)
        return

    # Combined Overlay
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for g in groups:
        _plot_group_curve_ax(
            ax=ax,
            data=valid_data,
            group=g,
            curve_type=curve_type,
            n_bins=n_bins,
            decimal_places=decimal_places,
            label_mode="full",
            curve_kwargs=final_curve_kwgs[g],
            line_kwargs=line_kwgs,
        )
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show_plot(fig, save_path, f"{filename}_overlay")


################################################################################
# Bootstrapped Helper for Group Curve Interpolation (All curve types)
################################################################################


def interpolate_bootstrapped_curves(
    boot_sliced_data, grid_x, curve_type="roc", n_bins=10
):
    """
    Interpolate bootstrapped ROC, PR, or calibration curves over a common x-axis grid.

    - boot_sliced_data: list of dicts per bootstrap sample with y_true and y_prob per group
    - grid_x: common x-axis values to interpolate onto (overridden for calibration)
    - curve_type: one of 'roc', 'pr', or 'calibration'
    - n_bins: number of bins for calibration curves
    """

    result = {}
    if curve_type == "calibration":
        bins = np.linspace(0, 1, n_bins + 1)
        grid_x = (bins[:-1] + bins[1:]) / 2  # override with bin centers

    for bootstrap_iter in boot_sliced_data:
        for group, values in bootstrap_iter.items():
            y_true = values["y_true"]
            y_prob = values["y_prob"]
            try:
                if curve_type == "roc":
                    x_vals, y_vals, _ = roc_curve(y_true, y_prob)
                    interp_func = interp1d(
                        x_vals, y_vals, bounds_error=False, fill_value=(0, 1)
                    )
                    y_interp = interp_func(grid_x)
                elif curve_type == "pr":
                    y_vals, x_vals, _ = precision_recall_curve(y_true, y_prob)
                    interp_func = interp1d(
                        x_vals, y_vals, bounds_error=False, fill_value=(0, 1)
                    )
                    y_interp = interp_func(grid_x)
                elif curve_type == "calibration":
                    y_interp = np.full(n_bins, np.nan)
                    for i in range(n_bins):
                        mask = (y_prob >= bins[i]) & (
                            (y_prob < bins[i + 1])
                            if i < n_bins - 1
                            else (y_prob <= bins[i + 1])
                        )
                        if np.any(mask):
                            y_interp[i] = np.mean(y_true[mask])
                else:
                    continue
            except Exception:
                y_interp = np.full_like(
                    grid_x if curve_type != "calibration" else np.arange(n_bins),
                    np.nan,
                )
            result.setdefault(group, []).append(y_interp)
    return result, grid_x


################################################################################
# Bootstrapped Group Curve Plot
################################################################################


def _plot_bootstrapped_curve_ax(
    ax,
    y_array,
    grid_x,
    group,
    label_prefix="AUROC",
    curve_kwargs=None,
    fill_kwargs=None,
    line_kwargs=None,
    show_grid=True,
    bar_every=10,
):
    """
    Plot mean curve with confidence band and error bars from bootstrapped
    interpolated data.
    """

    mean_y = np.nanmean(y_array, axis=0)
    lower = np.nanpercentile(y_array, 2.5, axis=0)
    upper = np.nanpercentile(y_array, 97.5, axis=0)
    aucs = (
        [np.trapz(y, grid_x) for y in y_array if not np.isnan(y).any()]
        if label_prefix != "CAL"
        else []
    )
    mean_auc = np.mean(aucs) if aucs else float("nan")
    lower_auc = np.percentile(aucs, 2.5) if aucs else float("nan")
    upper_auc = np.percentile(aucs, 97.5) if aucs else float("nan")

    label = (
        f"{group} ({label_prefix} = {mean_auc:.2f} [{lower_auc:.2f}, {upper_auc:.2f}])"
        if label_prefix != "CAL"
        else group
    )

    curve_kwargs = curve_kwargs or {}
    fill_kwargs = fill_kwargs or {
        "alpha": 0.2,
        "color": curve_kwargs.get("color", "#1f77b4"),
    }
    line_kwargs = line_kwargs or default_line_kwargs()

    ax.plot(grid_x, mean_y, label=label, **curve_kwargs)
    ax.fill_between(grid_x, lower, upper, **fill_kwargs)

    for j in range(0, len(grid_x), int(np.ceil(len(grid_x) / bar_every))):
        x_val = grid_x[j]
        mean_val = mean_y[j]
        err_low = max(mean_val - lower[j], 0)
        err_high = max(upper[j] - mean_val, 0)
        ax.errorbar(
            x_val,
            mean_val,
            yerr=[[err_low], [err_high]],
            fmt="o",
            color=curve_kwargs.get("color", "#1f77b4"),
            markersize=3,
            capsize=2,
            elinewidth=1,
            alpha=0.6,
        )

    if label_prefix in ["AUROC", "CAL"]:
        ax.plot([0, 1], [0, 1], **line_kwargs)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
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
    ax.tick_params(axis="both")
    ax.grid(show_grid)
    ax.legend(loc="lower right")


def eq_plot_bootstrapped_group_curves(
    boot_sliced_data,
    curve_type="roc",
    title="Bootstrapped Curve by Group",
    filename="bootstrapped_curve",
    save_path=None,
    dpi=100,
    figsize=(8, 6),
    common_grid=np.linspace(0, 1, 100),
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    color_by_group=True,
    curve_kwgs=None,
    fill_kwgs=None,
    line_kwgs=None,
    bar_every=10,
    n_bins=10,
):
    interp_data, grid_x = interpolate_bootstrapped_curves(
        boot_sliced_data, common_grid, curve_type=curve_type, n_bins=n_bins
    )

    group_names = sorted(interp_data.keys())

    color_map = (
        get_group_color_map(group_names)
        if color_by_group
        else {g: "#1f77b4" for g in group_names}
    )

    def get_kwargs(g):
        ck = {"color": color_map[g]}
        fk = {"alpha": 0.2, "color": color_map[g]}
        if curve_kwgs and g in curve_kwgs:
            ck.update(curve_kwgs[g])
        if fill_kwgs and g in fill_kwgs:
            fk.update(fill_kwgs[g])
        return ck, fk

    label_prefix = (
        "AUROC" if curve_type == "roc" else "AUCPR" if curve_type == "pr" else "CAL"
    )

    if group:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        valid_curves = [y for y in interp_data[group] if not np.isnan(y).all()]
        if not valid_curves:
            print(f"[Warning] Group '{group}' has no valid interpolated curves.")
            return
        y_array = np.vstack(valid_curves)
        ck, fk = get_kwargs(group)
        _plot_bootstrapped_curve_ax(
            ax,
            y_array,
            grid_x,
            group,
            label_prefix,
            ck,
            fk,
            line_kwgs,
            True,
            bar_every,
        )
        fig.suptitle(f"{title} ({group})")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_or_show_plot(fig, save_path, f"{filename}_{group}")
        return

    if subplots:
        if n_rows is None:
            n_rows = int(np.ceil(len(group_names) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), dpi=dpi
        )
        axes = axes.flatten()
        for i, g in enumerate(group_names):
            valid_curves = [y for y in interp_data[g] if not np.isnan(y).all()]
            if not valid_curves:
                print(
                    f"[Warning] Group '{g}' has no valid interpolated curves. Skipping."
                )
                axes[i].axis("off")
                continue
            y_array = np.vstack(valid_curves)
            ck, fk = get_kwargs(g)
            _plot_bootstrapped_curve_ax(
                axes[i],
                y_array,
                grid_x,
                g,
                label_prefix,
                ck,
                fk,
                line_kwgs,
                True,
                bar_every,
            )
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_or_show_plot(fig, save_path, filename)
        return

    # Combined Overlay
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for g in group_names:
        valid_curves = [y for y in interp_data[g] if not np.isnan(y).all()]
        if not valid_curves:
            print(f"[Warning] Group '{g}' has no valid interpolated curves. Skipping.")
            continue
        y_array = np.vstack(valid_curves)
        ck, fk = get_kwargs(g)
        _plot_bootstrapped_curve_ax(
            ax,
            y_array,
            grid_x,
            g,
            label_prefix,
            ck,
            fk,
            line_kwgs,
            True,
            bar_every,
        )
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show_plot(fig, save_path, f"{filename}_overlay")


################################################################################
# Disparity Metrics (Violin or Box Plots)
################################################################################


def eq_disparity_metrics_plot(
    dispa,
    metric_cols,
    name,
    plot_kind="violinplot",
    categories="all",
    include_legend=True,
    cmap="tab20c",
    color_by_group=True,
    save_path=None,
    filename="Disparity_Metrics",
    max_cols=None,
    strict_layout=True,
    figsize=None,
    **plot_kwargs,
):
    """
    Plot bootstrapped ROC, PR, or Calibration curves by group with confidence bands.

    - boot_sliced_data: list of dicts containing group-level bootstrap results
      with y_true and y_prob
    - curve_type: str, one of 'roc', 'pr', or 'calibration' to choose curve type
    - title: str, plot or grid title
    - filename: str, base filename for saving
    - save_path: optional directory to save output
    - dpi: resolution of figure
    - figsize: tuple, size of individual figure or subplot
    - common_grid: np.array, shared x-axis grid for interpolation
    - subplots: bool, whether to show per-group subplots
    - n_cols: number of columns in subplot grid
    - n_rows: optional override for subplot rows
    - group: str, optional single group to isolate and plot
    - color_by_group: bool, whether to color by group or use a uniform color
    - curve_kwgs: dict of group -> curve styling kwargs
    - fill_kwgs: dict of group -> fill_between styling kwargs
    - line_kwgs: kwargs for reference diagonal line
    - bar_every: int, controls interval for error bars
    - n_bins: int, number of bins used for calibration curve
    """

    if type(dispa) is not list:
        raise TypeError("dispa should be a list")

    all_keys = sorted({key for row in dispa for key in row.keys()})
    attributes = (
        [k for k in all_keys if k in categories] if categories != "all" else all_keys
    )

    color_map = plt.get_cmap(cmap)
    actual_colors = [color_map(i / len(attributes)) for i in range(len(attributes))]
    group_color_dict = {
        attr: (actual_colors[i] if color_by_group else "#1f77b4")
        for i, attr in enumerate(attributes)
    }

    # Always use original colors for legend and ticks
    legend_color_dict = {attr: actual_colors[i] for i, attr in enumerate(attributes)}

    legend_handles = [
        Line2D([0], [0], color=legend_color_dict[attr], lw=4, label=attr)
        for attr in attributes
    ]

    n_metrics = len(metric_cols)
    n_rows, n_cols, final_figsize = get_layout(
        n_metrics,
        n_cols=max_cols,
        figsize=figsize,
        strict_layout=strict_layout,
    )

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=final_figsize,
        squeeze=False,
    )

    for i, col in enumerate(metric_cols):
        ax = axs[i // n_cols, i % n_cols]
        x_vals = []
        y_vals = []
        for row in dispa:
            for attr in attributes:
                if attr in row:
                    x_vals.append(attr)
                    y_vals.append(row[attr][col])

        try:
            plot_func = getattr(sns, plot_kind)
        except AttributeError:
            raise ValueError(
                f"Unsupported plot_kind: '{plot_kind}'. Must be a seaborn plot type."
            )

        plot_func(
            ax=ax,
            x=x_vals,
            y=y_vals,
            hue=x_vals,
            palette=group_color_dict,
            legend=False,
            **plot_kwargs,
        )
        ax.set_title(f"{name}_{col}")
        ax.set_xlabel("")

        from matplotlib.ticker import FixedLocator

        locator = FixedLocator(list(range(len(attributes))))
        ax.xaxis.set_major_locator(locator)
        ax.set_xticks(list(range(len(attributes))))
        ax.set_xticklabels(attributes, rotation=0, fontweight="bold")

        for tick_label in ax.get_xticklabels():
            tick_label.set_color(legend_color_dict.get(tick_label.get_text(), "black"))
            tick_label.set_fontweight("bold")

        ax.hlines(
            [0, 1, 2], -1, len(attributes) + 1, ls=":", colors=["red", "black", "red"]
        )
        ax.set_xlim([-1, len(attributes)])
        ax.set_ylim(-2, 4)

    for j in range(i + 1, n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")

    if include_legend:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(attributes),
            fontsize="large",
            frameon=False,
        )

    plt.tight_layout(w_pad=2, h_pad=2, rect=[0.01, 0.01, 1.01, 1])
    save_or_show_plot(fig, save_path, f"{filename}_overlay")
