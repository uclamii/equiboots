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

from .metrics import regression_metrics

################################################################################
# Shared Utilities
################################################################################

DEFAULT_LINE_KWARGS = {"color": "black", "linestyle": "--", "linewidth": 1}
DEFAULT_LEGEND_KWARGS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, -0.25),
    "ncol": 1,
}


def save_or_show_plot(fig, save_path=None, filename="plot"):
    """Save plot to file if path is provided, otherwise display it."""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"{filename}.png"), bbox_inches="tight")
    plt.show()


def get_group_color_map(groups, palette="tab10"):
    """Generate a mapping from group names to colors."""
    colors = plt.get_cmap(palette).colors
    return {g: colors[i % len(colors)] for i, g in enumerate(groups)}


def get_layout(n_items, n_cols=None, figsize=None, strict_layout=True):
    """Compute layout grid and figure size based on number of items."""
    n_cols = n_cols or (6 if strict_layout else int(np.ceil(np.sqrt(n_items))))
    n_rows = int(np.ceil(n_items / n_cols))
    fig_width, fig_height = figsize or (
        (24, 4 * n_rows) if strict_layout else (5 * n_cols, 5 * n_rows)
    )
    return n_rows, n_cols, (fig_width, fig_height)


def _filter_groups(data, exclude_groups=0):
    """Filter out groups with one class or based on exclusion criteria."""
    valid_data = {g: v for g, v in data.items() if len(set(v["y_true"])) > 1}
    if not exclude_groups:  # If exclude_groups is 0 or None, return all valid data
        return valid_data

    # Handle case where exclude_groups is a specific group name (string) or list of names
    if isinstance(exclude_groups, (str, list, set)):
        exclude_set = (
            {exclude_groups} if isinstance(exclude_groups, str) else set(exclude_groups)
        )
        return {g: v for g, v in valid_data.items() if g not in exclude_set}

    # Handle case where exclude_groups is an integer (minimum sample size)
    if isinstance(exclude_groups, int):
        return {
            g: v for g, v in valid_data.items() if len(v["y_true"]) >= exclude_groups
        }

    raise ValueError("exclude_groups must be an int, str, list, or set")


def _get_concatenated_group_data(boot_sliced_data):
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


def plot_with_layout(
    data,
    plot_func,
    plot_kwargs,
    title="Plot",
    filename="plot",
    save_path=None,
    figsize=(8, 6),
    dpi=100,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    color_by_group=True,
    exclude_groups=0,
    show_grid=True,
):
    """Generic plotting wrapper for single-group, subplots, or overlay modes."""
    valid_data = _filter_groups(data, exclude_groups)
    groups = sorted(valid_data.keys())
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
            ax, valid_data, group, color_map[group], **plot_kwargs, overlay_mode=False
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
                axes[i], valid_data, g, color_map[g], **plot_kwargs, overlay_mode=False
            )
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for g in groups:
            plot_func(ax, valid_data, g, color_map[g], **plot_kwargs, overlay_mode=True)
        ax.set_title(title)
        ax.legend(**DEFAULT_LEGEND_KWARGS)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_or_show_plot(fig, save_path, filename)


################################################################################
# Residual Plot by Group
################################################################################


def _plot_residuals_ax(
    ax, y_true, y_pred, label, color, alpha=0.6, show_centroid=True, show_grid=True
):
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


def get_regression_label(y_true, y_pred, group):
    """Generate label with regression metrics."""
    metrics = regression_metrics(y_true, y_pred)
    return (
        f"R² for {group} = {metrics['R^2 Score']:.2f}, "
        f"MAE = {metrics['Mean Absolute Error']:.2f}, "
        f"Residual μ = {metrics['Residual Mean']:.2f}, "
        f"n = {len(y_true):,}"
    )


def eq_plot_residuals_by_group(
    data,
    alpha=0.6,
    show_centroids=False,
    title="Residuals by Group",
    filename="residuals_by_group",
    save_path=None,
    figsize=(8, 6),
    dpi=100,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    color_by_group=True,
    exclude_groups=0,
    show_grid=True,
):
    """Plot residuals grouped by subgroup."""
    if group and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    def residual_plot(
        ax, data, group, color, overlay_mode=False
    ):  # Added overlay_mode parameter
        y_true = data[group].get("y_true", data[group].get("y_actual"))
        y_pred = data[group].get("y_prob", data[group].get("y_pred"))
        label = get_regression_label(y_true, y_pred, group)
        _plot_residuals_ax(
            ax, y_true, y_pred, label, color, alpha, show_centroids, show_grid=show_grid
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
    ax,
    data,
    group,
    color,
    curve_type="roc",
    n_bins=10,
    decimal_places=2,
    label_mode="full",
    curve_kwargs=None,
    line_kwargs=None,
    show_legend=True,
    title=None,
    is_subplot=False,
    single_group=False,
    show_grid=True,
):
    """Plot ROC, PR, or calibration curve for a single group."""
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
    elif curve_type == "pr":
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        score = auc(recall, precision)
        x, y = recall, precision
        x_label, y_label = "Recall", "Precision"
        ref_line = ([0, 1], [positives / total] * 2)
        prefix = "AUCPR"
    elif curve_type == "calibration":
        y, x = calibration_curve(y_true, y_prob, n_bins=n_bins)
        score = brier_score_loss(y_true, y_prob)
        x_label, y_label = "Mean Predicted Value", "Fraction of Positives"
        ref_line = ([0, 1], [0, 1])
        prefix = "Brier score"
    else:
        raise ValueError("Unsupported curve_type")

    # Use the label_mode directly as passed from eq_plot_group_curves
    label = (
        f"{prefix} = {score:.{decimal_places}f}"
        if label_mode == "simple"
        else (
            f"{prefix} for {group} = {score:.{decimal_places}f}, "
            f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,}"
        )
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
        # Adjust legend position based on curve type and subplot/single group status
        if is_subplot or single_group:
            if curve_type == "roc":
                legend_kwargs = {"loc": "lower right"}
            elif curve_type == "pr":
                legend_kwargs = {"loc": "upper right"}
            elif curve_type == "calibration":
                legend_kwargs = {"loc": "lower right"}
            else:
                legend_kwargs = {"loc": "best"}
        else:
            legend_kwargs = DEFAULT_LEGEND_KWARGS
        ax.legend(**legend_kwargs)
    ax.grid(show_grid)
    ax.tick_params(axis="both")


def eq_plot_group_curves(
    data,
    curve_type="roc",
    n_bins=10,
    decimal_places=2,
    curve_kwgs=None,
    line_kwgs=None,
    title="Curve by Group",
    filename="group_curve",
    save_path=None,
    figsize=(8, 6),
    dpi=100,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    color_by_group=True,
    exclude_groups=0,
    show_grid=True,
):
    """Plot ROC, PR, or calibration curves by group."""
    if group and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

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
        data,
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
    boot_sliced_data, grid_x, curve_type="roc", n_bins=10
):
    """Interpolate bootstrapped curves over a common x-axis grid."""
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
            except Exception:
                y_interp = np.full_like(
                    grid_x if curve_type != "calibration" else np.arange(n_bins), np.nan
                )
            result.setdefault(group, []).append(y_interp)
    return result, grid_x


def _plot_bootstrapped_curve_ax(
    ax,
    y_array,
    grid_x,
    group,
    label_prefix="AUROC",
    curve_kwargs=None,
    fill_kwargs=None,
    line_kwargs=None,
    show_grid=True,  # Already has show_grid parameter
    bar_every=10,
    brier_scores=None,
):
    """Plot mean curve with confidence band and error bars."""
    mean_y = np.nanmean(y_array, axis=0)
    lower, upper = np.nanpercentile(y_array, [2.5, 97.5], axis=0)
    aucs = (
        [np.trapz(y, grid_x) for y in y_array if not np.isnan(y).all()]
        if label_prefix != "CAL"
        else []
    )
    mean_auc = np.mean(aucs) if aucs else float("nan")
    lower_auc, upper_auc = (
        np.percentile(aucs, [2.5, 97.5]) if aucs else (float("nan"), float("nan"))
    )

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

    curve_kwargs = curve_kwargs or {"color": "#1f77b4"}
    fill_kwargs = fill_kwargs or {
        "alpha": 0.2,
        "color": curve_kwargs.get("color", "#1f77b4"),
    }
    line_kwargs = line_kwargs or DEFAULT_LINE_KWARGS

    ax.plot(grid_x, mean_y, label=label, **curve_kwargs)
    ax.fill_between(grid_x, lower, upper, **fill_kwargs)
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
    ax.grid(show_grid)
    ax.legend(loc="lower right")


def eq_plot_bootstrapped_group_curves(
    boot_sliced_data,
    curve_type="roc",
    common_grid=np.linspace(0, 1, 100),
    bar_every=10,
    n_bins=10,
    title="Bootstrapped Curve by Group",
    filename="bootstrapped_curve",
    save_path=None,
    figsize=(8, 6),
    dpi=100,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    color_by_group=True,
    exclude_groups=0,
    show_grid=True,
):
    """Plot bootstrapped curves by group."""
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

    def boot_plot(
        ax, data, group, color, overlay_mode=False
    ):  # Added overlay_mode parameter
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
        group_data,
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
    show_grid=True,
    **plot_kwargs,
):
    """Plot disparity metrics as violin or box plots."""
    if not isinstance(dispa, list):
        raise TypeError("dispa should be a list")

    all_keys = sorted({key for row in dispa for key in row.keys()})
    attributes = (
        [k for k in all_keys if k in categories] if categories != "all" else all_keys
    )
    color_map = plt.get_cmap(cmap)
    colors = [color_map(i / len(attributes)) for i in range(len(attributes))]
    group_colors = {
        attr: (colors[i] if color_by_group else "#1f77b4")
        for i, attr in enumerate(attributes)
    }
    legend_colors = {attr: colors[i] for i, attr in enumerate(attributes)}

    n_rows, n_cols, figsize = get_layout(
        len(metric_cols), max_cols, figsize, strict_layout
    )
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i, col in enumerate(metric_cols):
        ax = axs[i // n_cols, i % n_cols]
        x_vals, y_vals = [], []
        for row in dispa:
            for attr in attributes:
                if attr in row:
                    x_vals.append(attr)
                    y_vals.append(row[attr][col])

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
            tick_label.set_color(legend_colors.get(tick_label.get_text(), "black"))
        ax.hlines(
            [0, 1, 2], -1, len(attributes) + 1, ls=":", colors=["red", "black", "red"]
        )
        ax.set_xlim(-1, len(attributes))
        ax.set_ylim(-2, 4)
        ax.grid(show_grid)

    for j in range(i + 1, n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")

    if include_legend:
        legend_handles = [
            Line2D([0], [0], color=legend_colors[attr], lw=4, label=attr)
            for attr in attributes
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(attributes),
            fontsize="large",
            frameon=False,
        )

    plt.tight_layout(w_pad=2, h_pad=2, rect=[0.01, 0.01, 1.01, 1])
    save_or_show_plot(fig, save_path, filename)
