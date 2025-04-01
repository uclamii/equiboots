from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
import os
import math

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)


################################################################################
# Save or Show Plots Utility Function
################################################################################


def save_or_show_plot(
    fig,
    save_path=None,
    filename="plot",
    bbox_inches="tight",
):
    """
    Save the plot to the specified path or show it if no path is provided.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save or show.

    save_path : str, optional
        Directory to save the plot. If None, the plot is shown instead.

    filename : str, optional
        Filename to use when saving the figure. Default is "plot".

    bbox_inches : str, optional
        Bounding box parameter for saving the figure. Default is "tight".

    Returns
    -------
    None
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename}.png"),
            bbox_inches=bbox_inches,
        )
    plt.show()


################################################################################
# Regression Residuals
################################################################################


def eq_plot_residuals_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group: np.ndarray,
    title: str = "Residuals by Group",
    filename: str = "residuals_by_group",
    save_path: str = None,
    figsize: tuple = (8, 6),
    dpi: int = 100,
    alpha: float = 0.6,
    tick_fontsize: int = 10,
    subplots: bool = False,
    color_by_group: bool = True,
    n_cols: int = 2,
    n_rows: int = None,
    show_centroids: bool = False,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    residuals = y_true - y_prob
    unique_groups = np.unique(group)

    palette = (
        sns.color_palette("tab10", len(unique_groups))
        if color_by_group
        else ["gray"] * len(unique_groups)
    )
    color_map = dict(zip(unique_groups, palette))

    def _scatter_with_centroid(ax, grp, mask, show_centroids=True):
        ax.scatter(
            y_prob[mask],
            residuals[mask],
            color=color_map[grp],
            alpha=alpha,
            label=str(grp),
        )
        if show_centroids:
            # Drop shadow
            ax.scatter(
                np.mean(y_prob[mask]) + 0.3,
                np.mean(residuals[mask]) - 0.3,
                color="black",
                marker="X",
                s=130,
                alpha=0.4,
                linewidth=0,
                zorder=4,
            )
            # Foreground marker
            ax.scatter(
                np.mean(y_prob[mask]),
                np.mean(residuals[mask]),
                color=color_map[grp],
                marker="X",
                s=120,
                edgecolor="black",
                linewidth=2,
                zorder=5,
                label=f"{grp} (centroid)" if not subplots else None,
            )

    if subplots:
        if n_rows is None:
            n_rows = int(np.ceil(len(unique_groups) / n_cols))
        elif n_rows * n_cols < len(unique_groups):
            print(
                f"[Warning] Grid size {n_rows}x{n_cols} only supports {n_rows * n_cols} plots; "
                f"showing first {n_rows * n_cols} of {len(unique_groups)} groups."
            )

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
        )
        axes = axes.flatten()

        for i, grp in enumerate(unique_groups):
            if i >= len(axes):
                break
            mask = group == grp
            ax = axes[i]
            _scatter_with_centroid(ax, grp, mask, show_centroids=show_centroids)

            ax.axhline(0, linestyle="--", color="gray", linewidth=1)
            ax.set_title(str(grp), fontsize=tick_fontsize + 1)
            ax.set_xlabel("Predicted Value", fontsize=tick_fontsize)
            ax.set_ylabel("Residual", fontsize=tick_fontsize)
            ax.grid(True)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=tick_fontsize + 3)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        legend_entries = []

        for grp in unique_groups:
            mask = group == grp
            _scatter_with_centroid(ax, grp, mask, show_centroids=show_centroids)

            y_true_grp = y_true[mask]
            y_prob_grp = y_prob[mask]
            residual_grp = y_true_grp - y_prob_grp

            r2 = r2_score(y_true_grp, y_prob_grp)
            mae = mean_absolute_error(y_true_grp, y_prob_grp)
            residual_mean = np.mean(residual_grp)
            n = len(y_true_grp)

            label = (
                f"R² for {grp} = {r2:.2f}, "
                f"MAE = {mae:.2f}, "
                f"Residual μ = {residual_mean:.2f}, "
                f"n = {n:,}"
            )

            legend_entries.append((label, color_map[grp]))

        # Format legend
        custom_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=color,
                markersize=8,
            )
            for label, color in legend_entries
        ]

        ax.legend(
            handles=custom_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fontsize=tick_fontsize,
            ncol=1,
            title="Group Stats",
        )

        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(title, fontsize=tick_fontsize + 2)
        ax.set_xlabel("Predicted Value", fontsize=tick_fontsize)
        ax.set_ylabel("Residual (y_true - y_prob)", fontsize=tick_fontsize)
        ax.grid(True)
        fig.tight_layout()

    save_or_show_plot(fig, save_path=save_path, filename=filename)


################################################################################
# ROC AUC Curve Plot
################################################################################


def _plot_single_roc_ax(
    ax,
    data: dict,
    group: str,
    title=None,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    decimal_places=2,
    show_legend=True,
    label_mode="full",
    curve_kwargs=None,
    line_kwargs=None,
):
    """
    Helper to extract data for `group` and plot ROC curve on the given axis.
    """
    y_true = data[group]["y_true"]
    y_prob = data[group]["y_prob"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    total = len(y_true)
    positives = int(np.sum(y_true))
    negatives = total - positives

    if label_mode == "simple":
        label = f"AUC = {roc_auc:.{decimal_places}f}"
    else:
        label = (
            f"AUC for {group} = {roc_auc:.{decimal_places}f}, "
            f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,}"
        )

    ax.plot(fpr, tpr, label=label, **(curve_kwargs or {}))
    ax.plot(
        [0, 1],
        [0, 1],
        **(line_kwargs or {"color": "k", "linestyle": "--", "linewidth": 1}),
    )
    if title:
        ax.set_title(title, fontsize=label_fontsize)
    ax.set_xlabel("False Positive Rate", fontsize=label_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=label_fontsize)
    if show_legend:
        ax.legend(fontsize=tick_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    ax.set_xlabel("False Positive Rate", fontsize=label_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=label_fontsize)
    if show_legend:
        ax.legend(fontsize=tick_fontsize)
    ax.grid(True)


def eq_plot_roc_auc(
    data: dict,
    save_path: str = None,
    filename: str = "roc_auc_by_group",
    title: str = "ROC Curve by Group",
    figsize: tuple = (8, 6),
    dpi: int = 100,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    decimal_places: int = 2,
    subplots: bool = False,
    n_cols: int = 2,
    n_rows: int = None,
    group: str = None,
    color_by_group: bool = True,
    curve_kwgs: dict = None,
    line_kwgs: dict = None,
):
    """
    Plots ROC AUC curves for each group in a fairness dictionary, with support
    for single group, all groups, or subplot layout. Allows custom styling for curves and discrimination line.
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    if title == None:
        title = ""

    valid_data = {g: v for g, v in data.items() if len(set(v["y_true"])) > 1}
    groups = sorted(valid_data.keys())

    # Default 45° discrimination line style
    if line_kwgs is None:
        line_kwgs = {"color": "k", "linestyle": "--", "linewidth": 1}

    # Default color assignment
    default_colors = {}
    if color_by_group:
        palette = plt.get_cmap("tab10").colors
        default_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    # Merge default colors and user-defined kwargs
    final_curve_kwgs = {}
    for g in groups:
        final_curve_kwgs[g] = {"color": default_colors.get(g, "gray")}
        if curve_kwgs and g in curve_kwgs:
            final_curve_kwgs[g].update(curve_kwgs[g])

    # Plot for a single group
    if group:
        if group not in valid_data:
            print(
                f"[Warning] Group '{group}' not found or insufficient class diversity."
            )
            return

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot_single_roc_ax(
            ax=ax,
            data=valid_data,
            group=group,
            title=f"{title} ({group})",
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            decimal_places=decimal_places,
            show_legend=True,
            label_mode="simple",
            curve_kwargs=final_curve_kwgs[group],
            line_kwargs=line_kwgs,
        )
        fig.tight_layout()
        save_or_show_plot(fig, save_path=save_path, filename=f"{filename}_{group}")
        return

    # Subplots
    if subplots:
        if n_rows is None:
            n_rows = int(np.ceil(len(groups) / n_cols))
        elif n_rows * n_cols < len(groups):
            print(
                f"[Warning] Grid size {n_rows}x{n_cols} only supports {n_rows * n_cols} plots; "
                f"showing first {n_rows * n_cols} of {len(groups)} groups."
            )

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, dpi=dpi)
        axes = np.atleast_1d(axes).flatten()

        for i, g in enumerate(groups):
            if i >= len(axes):
                break
            _plot_single_roc_ax(
                ax=axes[i],
                data=valid_data,
                group=g,
                title=str(g),
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                decimal_places=decimal_places,
                show_legend=True,
                label_mode="simple",
                curve_kwargs=final_curve_kwgs[g],
                line_kwargs=line_kwgs,
            )

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=tick_fontsize + 3)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Combined overlay
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for g in groups:
            _plot_single_roc_ax(
                ax=ax,
                data=valid_data,
                group=g,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                decimal_places=decimal_places,
                show_legend=False,
                label_mode="full",
                curve_kwargs=final_curve_kwgs[g],
                line_kwargs=line_kwgs,
            )

        ax.set_title(title, fontsize=tick_fontsize + 2)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fontsize=tick_fontsize,
            ncol=1,
        )
        fig.tight_layout()

    save_or_show_plot(fig, save_path=save_path, filename=filename)


################################################################################
# Precision-Recall Curve Plot
################################################################################


def _plot_single_pr_ax(
    ax,
    data: dict,
    group: str,
    title=None,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    decimal_places=2,
    show_legend=True,
    label_mode="full",
    curve_kwargs=None,
    line_kwargs=None,
):
    """
    Helper to extract data for `group` and plot PR curve on the given axis.
    """
    y_true = data[group]["y_true"]
    y_prob = data[group]["y_prob"]

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    total = len(y_true)
    positives = int(np.sum(y_true))
    negatives = total - positives

    if label_mode == "simple":
        label = f"AP = {avg_precision:.{decimal_places}f}"
    else:
        label = (
            f"AP for {group} = {avg_precision:.{decimal_places}f}, "
            f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,}"
        )

    ax.plot(recall, precision, label=label, **(curve_kwargs or {}))
    ax.axhline(
        y=positives / total,
        **(line_kwargs or {"color": "k", "linestyle": "--", "linewidth": 1}),
    )
    if title:
        ax.set_title(title, fontsize=label_fontsize)
    ax.set_xlabel("Recall", fontsize=label_fontsize)
    ax.set_ylabel("Precision", fontsize=label_fontsize)
    if show_legend:
        ax.legend(fontsize=tick_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_xlabel("Recall", fontsize=label_fontsize)
    ax.set_ylabel("Precision", fontsize=label_fontsize)
    if show_legend:
        ax.legend(fontsize=tick_fontsize)
    ax.grid(True)


def eq_plot_precision_recall(
    data: dict,
    save_path: str = None,
    filename: str = "precision_recall_by_group",
    title: str = "Precision-Recall Curve by Group",
    figsize: tuple = (8, 6),
    dpi: int = 100,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    decimal_places: int = 2,
    subplots: bool = False,
    n_cols: int = 2,
    n_rows: int = None,
    group: str = None,
    color_by_group: bool = True,
    curve_kwgs: dict = None,
    line_kwgs: dict = None,
):
    """
    Plots Precision-Recall curves for each group in a fairness dictionary.

    Supports: single group, overlay, or subplot layout with full styling options.
    """
    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    valid_data = {g: v for g, v in data.items() if len(set(v["y_true"])) > 1}
    groups = sorted(valid_data.keys())

    if title == None:
        title = ""

    if line_kwgs is None:
        line_kwgs = {"color": "k", "linestyle": "--", "linewidth": 1}

    # Default color assignment
    default_colors = {}
    if color_by_group:
        palette = plt.get_cmap("tab10").colors
        default_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    # Merge defaults with user-specified curve_kwgs
    final_curve_kwgs = {}
    for g in groups:
        final_curve_kwgs[g] = {"color": default_colors.get(g, "gray")}
        if curve_kwgs and g in curve_kwgs:
            final_curve_kwgs[g].update(curve_kwgs[g])

    # Single group plot
    if group:
        if group not in valid_data:
            print(
                f"[Warning] Group '{group}' not found or insufficient class diversity."
            )
            return

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot_single_pr_ax(
            ax=ax,
            data=valid_data,
            group=group,
            title=f"{title} ({group})",
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            decimal_places=decimal_places,
            show_legend=True,
            label_mode="simple",
            curve_kwargs=final_curve_kwgs[group],
            line_kwargs=line_kwgs,
        )
        fig.tight_layout()
        save_or_show_plot(fig, save_path=save_path, filename=f"{filename}_{group}")
        return

    # Subplots
    if subplots:
        if n_rows is None:
            n_rows = int(np.ceil(len(groups) / n_cols))
        elif n_rows * n_cols < len(groups):
            print(
                f"[Warning] Grid size {n_rows}x{n_cols} only supports {n_rows * n_cols} plots; "
                f"showing first {n_rows * n_cols} of {len(groups)} groups."
            )

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, dpi=dpi)
        axes = np.atleast_1d(axes).flatten()

        for i, g in enumerate(groups):
            if i >= len(axes):
                break
            _plot_single_pr_ax(
                ax=axes[i],
                data=valid_data,
                group=g,
                title=str(g),
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                decimal_places=decimal_places,
                show_legend=True,
                label_mode="simple",
                curve_kwargs=final_curve_kwgs[g],
                line_kwargs=line_kwgs,
            )

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=tick_fontsize + 3)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Combined overlay
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for g in groups:
            _plot_single_pr_ax(
                ax=ax,
                data=valid_data,
                group=g,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                decimal_places=decimal_places,
                show_legend=False,
                label_mode="full",
                curve_kwargs=final_curve_kwgs[g],
                line_kwargs=line_kwgs,
            )

        ax.set_title(title, fontsize=tick_fontsize + 2)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fontsize=tick_fontsize,
            ncol=1,
        )
        fig.tight_layout()

    save_or_show_plot(fig, save_path=save_path, filename=filename)


################################################################################
# Calibration Curve Plot
################################################################################


def _plot_single_calibration_ax(
    ax,
    data: dict,
    group: str,
    n_bins: int,
    title=None,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    decimal_places: int = 2,
    show_legend=True,
    label_mode="full",
    curve_kwargs=None,
    line_kwargs=None,
):
    y_true = data[group]["y_true"]
    y_prob = data[group]["y_prob"]

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    brier = brier_score_loss(y_true, y_prob)
    total = len(y_true)
    positives = int(np.sum(y_true))
    negatives = total - positives

    if label_mode == "simple":
        label = f"Brier = {brier:.{decimal_places}f}"
    else:
        label = (
            f"{group} (Brier = {brier:.{decimal_places}f}, "
            f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,})"
        )

    ax.plot(
        np.round(mean_predicted_value, decimal_places),
        np.round(fraction_of_positives, decimal_places),
        marker="o",
        label=label,
        **(curve_kwargs or {}),
    )
    ax.plot([0, 1], [0, 1], **(line_kwargs or {"color": "gray", "linestyle": "--"}))

    if title:
        ax.set_title(title, fontsize=label_fontsize)
    ax.set_xlabel("Mean Predicted Value", fontsize=label_fontsize)
    ax.set_ylabel("Fraction of Positives", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    if show_legend:
        ax.legend(fontsize=tick_fontsize)
    ax.grid(True)


def eq_calibration_curve_plot(
    data: dict,
    n_bins: int = 10,
    save_path: str = None,
    filename: str = "calibration_by_group",
    title: str = "Calibration Curve by Group",
    figsize: tuple = (8, 6),
    dpi: int = 100,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    decimal_places: int = 2,
    subplots: bool = False,
    n_cols: int = 2,
    n_rows: int = None,
    group: str = None,
    color_by_group: bool = True,
    curve_kwgs: dict = None,
    line_kwgs: dict = None,
):
    """
    Plots calibration curves for each group in a fairness dictionary.

    Parameters match ROC and PR curve plotters for consistency.
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    if title == None:
        title = ""

    valid_data = {g: v for g, v in data.items() if len(set(v["y_true"])) > 1}
    groups = sorted(valid_data.keys())

    if line_kwgs is None:
        line_kwgs = {"color": "gray", "linestyle": "--", "linewidth": 1}

    # Color handling
    default_colors = {}
    if color_by_group:
        palette = plt.get_cmap("tab10").colors
        default_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    final_curve_kwgs = {}
    for g in groups:
        final_curve_kwgs[g] = {"color": default_colors.get(g, "gray")}
        if curve_kwgs and g in curve_kwgs:
            final_curve_kwgs[g].update(curve_kwgs[g])

    # Single group mode
    if group:
        if group not in valid_data:
            print(
                f"[Warning] Group '{group}' not found or insufficient class diversity."
            )
            return
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot_single_calibration_ax(
            ax=ax,
            data=valid_data,
            group=group,
            n_bins=n_bins,
            title=title,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            decimal_places=decimal_places,
            show_legend=True,
            label_mode="simple",
            curve_kwargs=final_curve_kwgs[group],
            line_kwargs=line_kwgs,
        )
        fig.tight_layout()
        save_or_show_plot(fig, save_path=save_path, filename=f"{filename}_{group}")
        return

    # Subplots
    if subplots:
        if n_rows is None:
            n_rows = int(np.ceil(len(groups) / n_cols))
        elif n_rows * n_cols < len(groups):
            print(
                f"[Warning] Grid size {n_rows}x{n_cols} only supports {n_rows * n_cols} plots; "
                f"showing first {n_rows * n_cols} of {len(groups)} groups."
            )

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, dpi=dpi)
        axes = np.atleast_1d(axes).flatten()

        for i, g in enumerate(groups):
            if i >= len(axes):
                break
            _plot_single_calibration_ax(
                ax=axes[i],
                data=valid_data,
                group=g,
                n_bins=n_bins,
                title=str(g),
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                decimal_places=decimal_places,
                show_legend=True,
                label_mode="simple",
                curve_kwargs=final_curve_kwgs[g],
                line_kwargs=line_kwgs,
            )

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=label_fontsize + 2)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Combined overlay
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for g in groups:
            _plot_single_calibration_ax(
                ax=ax,
                data=valid_data,
                group=g,
                n_bins=n_bins,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                decimal_places=decimal_places,
                show_legend=False,
                label_mode="full",
                curve_kwargs=final_curve_kwgs[g],
                line_kwargs=line_kwgs,
            )

        ax.set_title(title, fontsize=label_fontsize)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fontsize=tick_fontsize,
            ncol=1,
        )
        fig.tight_layout()

    save_or_show_plot(fig, save_path=save_path, filename=filename)


################################################################################
# Disparity Metrics (Violin or Box Plots)
################################################################################


def get_layout(n_metrics, max_cols=None, figsize=None, strict_layout=True):
    if strict_layout:
        if max_cols is None:
            max_cols = 6  # fallback default
        n_cols = max_cols
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig_width = 24 if figsize is None else figsize[0]
        fig_height = 4 * n_rows if figsize is None else figsize[1]
    else:
        if figsize is not None:
            if max_cols is None:
                max_cols = int(np.ceil(np.sqrt(n_metrics)))
            n_cols = min(max_cols, n_metrics)
            n_rows = int(np.ceil(n_metrics / n_cols))
            fig_width, fig_height = figsize
        else:
            if max_cols is None:
                max_cols = int(np.ceil(np.sqrt(n_metrics)))
            n_cols = min(max_cols, n_metrics)
            n_rows = int(np.ceil(n_metrics / n_cols))
            fig_width = 5 * n_cols
            fig_height = 5 * n_rows

    return n_rows, n_cols, (fig_width, fig_height)


def eq_disparity_metrics_plot(
    dispa,
    metric_cols,
    name,
    plot_kind="violinplot",
    categories="all",
    include_legend=True,
    cmap="tab20c",
    save_path=None,
    filename="Disparity_Metrics",
    max_cols=None,
    strict_layout=True,
    figsize=None,
    **plot_kwargs,
):
    # Ensure necessary columns are in the DataFrame
    if type(dispa) is not list:
        raise TypeError("dispa should be a list")

    # Filter the DataFrame based on the specified categories
    if categories != "all":
        attributes = categories
    else:
        attributes = list(dispa[0].keys())

    # Create a dictionary to map attribute_value to labels A, B, C, etc.
    value_labels = {value: chr(65 + i) for i, value in enumerate(attributes)}

    # Reverse the dictionary to use in plotting
    label_values = {v: k for k, v in value_labels.items()}

    # Use a color map to generate colors
    color_map = plt.get_cmap(cmap)  # Allow user to specify colormap
    num_colors = len(label_values)
    colors = [color_map(i / num_colors) for i in range(num_colors)]

    # Create custom legend handles
    legend_handles = [
        plt.Line2D([0], [0], color=colors[j], lw=4, label=f"{label} = {value}")
        for j, (label, value) in enumerate(label_values.items())
    ]

    n_metrics = len(metric_cols)
    n_rows, n_cols, final_figsize = get_layout(
        n_metrics,
        max_cols=max_cols,
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
            for key, val in row.items():
                x_vals.append(key)
                y_vals.append(val[col])

        # Validate and get the Seaborn plotting function
        try:
            plot_func = getattr(sns, plot_kind)
        except AttributeError:
            raise ValueError(
                f"Unsupported plot_kind: '{plot_kind}'. Must be one of: "
                "'violinplot', 'boxplot', 'stripplot', 'swarmplot', etc."
            )
        plot_color = colors[0]
        plot_func(ax=ax, x=x_vals, y=y_vals, color=plot_color, **plot_kwargs)
        ax.set_title(name + "_" + col)
        ax.set_xlabel("")
        ax.set_xticks(range(len(label_values)))
        ax.set_xticklabels(
            label_values.keys(),
            rotation=0,
            fontweight="bold",
        )

        # Set the color and font weight of each tick label to match the
        # corresponding color in the legend
        for tick_label in ax.get_xticklabels():
            tick_label.set_color(
                colors[list(label_values.keys()).index(tick_label.get_text())]
            )
            tick_label.set_fontweight("bold")

        ax.hlines(0, -1, len(value_labels) + 1, ls=":", color="red")
        ax.hlines(1, -1, len(value_labels) + 1, ls=":")
        ax.hlines(2, -1, len(value_labels) + 1, ls=":", color="red")
        ax.set_xlim([-1, len(value_labels)])
        ax.set_ylim(-2, 4)

    # Keep empty axes but hide them (preserves layout spacing)
    for j in range(i + 1, n_rows * n_cols):
        ax = axs[j // n_cols, j % n_cols]
        ax.axis("off")

    # Before showing or saving
    if include_legend:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(label_values),
            fontsize="large",
            frameon=False,
        )

    plt.tight_layout(
        w_pad=2, h_pad=2, rect=[0.01, 0.01, 1.01, 1]
    )  # Adjust rect to make space for the legend and reduce white space

    save_or_show_plot(fig, save_path=save_path, filename=filename)


################################################################################
# Bootstrapped ROC AUC Curve Plot
################################################################################


def eq_plot_bootstrapped_roc_curves(
    boot_sliced_data,
    title="Bootstrapped ROC Curves by Group",
    filename="roc_curves_by_group_grid",
    save_path=None,
    dpi=100,
    figsize_per_plot=(6, 5),
    common_grid=np.linspace(0, 1, 100),
    alpha_fill=0.2,
    color="#1f77b4",
    bar_every=10,
):
    """
    Plot bootstrapped ROC curves with shaded confidence intervals,
    one group per subplot (grid layout).

    Parameters
    ----------
    boot_sliced_data : list of dicts
        Output of EquiBoots.slicer() with bootstrap_flag=True.
    common_grid : np.ndarray
        Common FPR grid to interpolate TPRs across bootstraps.
    figsize_per_plot : tuple
        Size (w, h) of each subplot.
    """
    group_fpr_tpr = {}

    for bootstrap_iter in boot_sliced_data:
        for group, values in bootstrap_iter.items():
            y_true = values["y_true"]
            y_prob = values["y_prob"]

            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                interp = interp1d(
                    fpr,
                    tpr,
                    bounds_error=False,
                    fill_value=(0, 1),
                )
                tpr_interp = interp(common_grid)
            except ValueError:
                tpr_interp = np.full_like(common_grid, np.nan)

            if group not in group_fpr_tpr:
                group_fpr_tpr[group] = []

            group_fpr_tpr[group].append(tpr_interp)

    group_names = sorted(group_fpr_tpr.keys())
    num_groups = len(group_names)
    n_cols = 2
    n_rows = math.ceil(num_groups / n_cols)
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        dpi=dpi,
    )
    axes = axes.flatten()

    for i, group in enumerate(group_names):
        ax = axes[i]
        tpr_array = np.vstack(
            [tpr for tpr in group_fpr_tpr[group] if not np.isnan(tpr).any()]
        )
        if tpr_array.shape[0] == 0:
            continue

        mean_tpr = np.mean(tpr_array, axis=0)
        lower = np.percentile(tpr_array, 2.5, axis=0)
        upper = np.percentile(tpr_array, 97.5, axis=0)
        aucs = [np.trapz(tpr, common_grid) for tpr in tpr_array]
        mean_auc = np.mean(aucs)
        lower_auc = np.percentile(aucs, 2.5)
        upper_auc = np.percentile(aucs, 97.5)
        auc_str = f"Mean AUROC = {mean_auc:.2f} [{lower_auc:.2f}, {upper_auc:.2f}]"

        ax.plot(common_grid, mean_tpr, label=auc_str, color=color)
        ax.fill_between(
            common_grid,
            lower,
            upper,
            alpha=alpha_fill,
            color=color,
        )

        for j in range(0, len(common_grid), int(np.ceil(len(common_grid) / bar_every))):
            fpr_val = common_grid[j]
            mean_val = mean_tpr[j]
            err_low = mean_val - lower[j]
            err_high = upper[j] - mean_val

            ax.errorbar(
                fpr_val,
                mean_val,
                yerr=[[err_low], [err_high]],
                fmt="o",
                color=color,
                markersize=3,
                capsize=2,
                elinewidth=1,
                alpha=0.6,
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(group, fontsize=12)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_or_show_plot(fig, save_path=save_path, filename=filename)


################################################################################
# Bootstrapped Precision-Recall Curve Plot
################################################################################


def eq_plot_bootstrapped_pr_curves(
    boot_sliced_data,
    title="Bootstrapped PR Curves by Group",
    filename="roc_curves_by_group_grid",
    save_path=None,
    dpi=100,
    figsize_per_plot=(6, 5),
    common_grid=np.linspace(0, 1, 100),
    alpha_fill=0.2,
    color="#1f77b4",
    bar_every=10,
):
    """
    Plot bootstrapped ROC curves with shaded confidence intervals,
    one group per subplot (grid layout).

    Parameters
    ----------
    boot_sliced_data : list of dicts
        Output of EquiBoots.slicer() with bootstrap_flag=True.
    common_grid : np.ndarray
        Common FPR grid to interpolate TPRs across bootstraps.
    figsize_per_plot : tuple
        Size (w, h) of each subplot.
    """
    group_pr = {}

    for bootstrap_iter in boot_sliced_data:
        for group, values in bootstrap_iter.items():
            y_true = values["y_true"]
            y_prob = values["y_prob"]

            try:
                precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
                interp = interp1d(
                    recalls,
                    precisions,
                    bounds_error=False,
                    fill_value=(0, 1),
                )
                precision_interp_func = interp(common_grid)
            except ValueError:
                precision_interp_func = np.full_like(common_grid, np.nan)

            if group not in group_pr:
                group_pr[group] = []

            group_pr[group].append(precision_interp_func)

    group_names = sorted(group_pr.keys())
    num_groups = len(group_names)
    n_cols = 2
    n_rows = math.ceil(num_groups / n_cols)
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        dpi=dpi,
    )
    axes = axes.flatten()

    for i, group in enumerate(group_names):
        ax = axes[i]
        precision_array = np.vstack(
            [
                precision
                for precision in group_pr[group]
                if not np.isnan(precision).any()
            ]
        )
        if precision_array.shape[0] == 0:
            continue

        mean_precision = np.mean(precision_array, axis=0)
        lower = np.percentile(precision_array, 2.5, axis=0)
        upper = np.percentile(precision_array, 97.5, axis=0)
        aucs = [np.trapz(tpr, common_grid) for tpr in precision_array]
        mean_auc = np.mean(aucs)
        lower_auc = np.percentile(aucs, 2.5)
        upper_auc = np.percentile(aucs, 97.5)
        auc_str = f"Mean AUCPR = {mean_auc:.2f} [{lower_auc:.2f}, {upper_auc:.2f}]"

        ax.plot(common_grid, mean_precision, label=auc_str, color=color)
        ax.fill_between(
            common_grid,
            lower,
            upper,
            alpha=alpha_fill,
            color=color,
        )

        for j in range(0, len(common_grid), int(np.ceil(len(common_grid) / bar_every))):
            fpr_val = common_grid[j]
            mean_val = mean_precision[j]
            err_low = mean_val - lower[j]
            err_high = upper[j] - mean_val

            ax.errorbar(
                fpr_val,
                mean_val,
                yerr=[[err_low], [err_high]],
                fmt="o",
                color=color,
                markersize=3,
                capsize=2,
                elinewidth=1,
                alpha=0.6,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(group, fontsize=12)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower right", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_or_show_plot(fig, save_path=save_path, filename=filename)


################################################################################
# Bootstrapped Calibration Curve Plot
################################################################################


def eq_plot_bootstrapped_calibration_curves(
    boot_sliced_data,
    title="Bootstrapped Calibration Curves by Group",
    filename="calibration_curves_by_group_grid",
    save_path=None,
    dpi=100,
    figsize_per_plot=(6, 5),
    n_bins=10,
    alpha_fill=0.2,
    color="#1f77b4",
    decimal_places=2,
):
    """
    Plot bootstrapped calibration curves (fraction of positives vs. predicted
    probability) with shaded confidence intervals, one group per subplot
    (grid layout). The curves are computed using fixed bins so that they remain
    jagged (i.e., not smoothed).

    Parameters
    ----------
    boot_sliced_data : list of dict
        Each element in the list represents one bootstrap iteration.
        Each element is a dictionary of the form:
            {
              "groupA": {"y_true": np.array([...]), "y_prob": np.array([...])},
              "groupB": {"y_true": np.array([...]), "y_prob": np.array([...])},
              ...
            }
    title : str
        Plot title.
    filename : str
        Name of the file to save (without extension).
    save_path : str or None
        Directory to save the plot. If None, the plot is shown instead of saved.
    dpi : int
        Dots per inch (plot resolution).
    figsize_per_plot : tuple
        Size (width, height) for each subplot.
    n_bins : int
        Number of bins to use for the calibration curve.
    alpha_fill : float
        Alpha (transparency) for the confidence interval shading.
    color : str
        Color for the main curve and shading.
    n_bars : int
        Approximate number of points at which to plot error bars.
    decimal_places : int
        Decimal precision for displayed metrics (e.g., Brier scores).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the subplots.
    """
    # Create fixed bin edges and corresponding bin centers
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Dictionaries to hold calibration curves and Brier scores for each group
    group_cal = (
        {}
    )  # Will store an array of fraction_of_positives per bootstrap iteration
    group_brier = {}  # Will store the Brier score per bootstrap iteration

    # 1. Gather calibration data for each bootstrap iteration
    for bootstrap_iter in boot_sliced_data:
        for group, values in bootstrap_iter.items():
            y_true = values["y_true"]
            y_prob = values["y_prob"]

            # Compute fraction of positives in each fixed bin
            frac_positives = np.empty(n_bins)
            frac_positives[:] = np.nan  # initialize as NaN
            for i in range(n_bins):
                # Use [bins[i], bins[i+1]) except for the last bin, which includes 1.
                if i < n_bins - 1:
                    bin_mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
                else:
                    bin_mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
                if np.any(bin_mask):
                    frac_positives[i] = np.mean(y_true[bin_mask])
                else:
                    frac_positives[i] = np.nan

            # Store the calibration curve (jagged, per fixed bins)
            group_cal.setdefault(group, []).append(frac_positives)

            # Compute and store the Brier score for this iteration
            brier = brier_score_loss(y_true, y_prob)
            group_brier.setdefault(group, []).append(brier)

    # 2. Create the subplot grid
    group_names = sorted(group_cal.keys())
    num_groups = len(group_names)
    n_cols = 2
    n_rows = math.ceil(num_groups / n_cols)
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # 3. Plot each group's calibration curves
    for i, group in enumerate(group_names):
        ax = axes[i]

        # Convert list of curves to a NumPy array: shape (num_bootstraps, n_bins)
        cal_array = np.array(group_cal[group])
        # Remove bootstrap iterations where all values are NaN
        valid_rows = ~np.all(np.isnan(cal_array), axis=1)
        cal_array = cal_array[valid_rows, :]
        if cal_array.shape[0] == 0:
            ax.set_title(group, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.plot([0, 1], [0, 1], "--", color="gray")
            continue

        # Compute mean calibration and 95% confidence bounds across bootstraps
        mean_cal = np.nanmean(cal_array, axis=0)
        lower_cal = np.nanpercentile(cal_array, 2.5, axis=0)
        upper_cal = np.nanpercentile(cal_array, 97.5, axis=0)

        # Compute Brier score statistics
        briers = np.array(group_brier[group])
        mean_brier = np.mean(briers)
        lower_brier = np.percentile(briers, 2.5)
        upper_brier = np.percentile(briers, 97.5)
        brier_str = (
            f"Mean Brier = {mean_brier:.{decimal_places}f} "
            f"[{lower_brier:.{decimal_places}f}, {upper_brier:.{decimal_places}f}]"
        )

        # Plot the mean calibration curve (using bin centers) with markers to show
        # jagged steps
        ax.plot(
            bin_centers,
            mean_cal,
            label=brier_str,
            color=color,
            marker="o",
        )

        # Shade the confidence interval
        ax.fill_between(
            bin_centers, lower_cal, upper_cal, alpha=alpha_fill, color=color
        )

        # Optionally, add error bars at a subset of points
        selected_indices = np.linspace(0, n_bins - 1, 10, dtype=int)
        for j in selected_indices:
            x_val = bin_centers[j]
            mean_val = mean_cal[j]
            err_low = mean_val - lower_cal[j]
            err_high = upper_cal[j] - mean_val
            ax.errorbar(
                x_val,
                mean_val,
                yerr=[[err_low], [err_high]],
                fmt="o",
                color=color,
                markersize=3,
                capsize=2,
                elinewidth=1,
                alpha=0.6,
            )  # Plot the diagonal for perfect calibration
        ax.plot([0, 1], [0, 1], "--", color="gray")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(group, fontsize=12)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True)

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # 4. Final formatting and save/show the figure
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_or_show_plot(fig, save_path=save_path, filename=filename)


def extract_group_metrics(race_metrics):
    unique_groups = set()
    for sample in race_metrics:
        unique_groups.update(sample.keys())

    metrics = {group: {"TPR": [], "FPR": []} for group in unique_groups}
    for sample in race_metrics:
        for group in unique_groups:
            metrics[group]["TPR"].append(sample[group].get("TP Rate"))
            metrics[group]["FPR"].append(sample[group].get("FP Rate"))
    return metrics, unique_groups


def compute_confidence_intervals(metrics, conf=95):
    conf_intervals = {}
    lower_percentile = (100 - conf) / 2
    upper_percentile = 100 - lower_percentile
    for group, group_metrics in metrics.items():
        conf_intervals[group] = {}
        for metric_name, values in group_metrics.items():
            values_clean = [v for v in values if v is not None]
            if values_clean:
                lower_bound = np.percentile(values_clean, lower_percentile)
                upper_bound = np.percentile(values_clean, upper_percentile)
                conf_intervals[group][metric_name] = (lower_bound, upper_bound)
            else:
                conf_intervals[group][metric_name] = (None, None)
    return conf_intervals
