from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
import os

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


def _plot_single_residual_ax(
    ax,
    y_true,
    y_prob,
    group_label,
    label_fontsize=12,
    tick_fontsize=10,
    alpha=0.6,
    color="tab:blue",
    line_kwargs=None,
    show_centroid=True,
):
    """
    Helper function to plot residuals vs predicted values for a single group.

    Used within grouped residual plotting routines to modularize logic, apply
    consistent styling, and support optional centroid visualization for
    interpretability.
    """

    residuals = y_true - y_prob

    line_kwargs = line_kwargs or {"linestyle": "--", "color": "gray", "linewidth": 1}

    ax.scatter(y_prob, residuals, alpha=alpha, label=str(group_label), color=color)

    if show_centroid:
        ax.scatter(
            np.mean(y_prob) + 0.3,
            np.mean(residuals) - 0.3,
            color="black",
            marker="X",
            s=130,
            alpha=0.4,
            linewidth=0,
            zorder=4,
        )
        ax.scatter(
            np.mean(y_prob),
            np.mean(residuals),
            color=color,
            marker="X",
            s=120,
            edgecolor="black",
            linewidth=2,
            zorder=5,
        )

    ax.axhline(0, **line_kwargs)
    ax.set_title(str(group_label), fontsize=label_fontsize)
    ax.set_xlabel("Predicted Value", fontsize=label_fontsize)
    ax.set_ylabel("Residual (y_true - y_prob)", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True)


def eq_plot_residuals_by_group(
    data: dict,
    save_path: str = None,
    filename: str = "residuals_by_group",
    title: str = "Residuals by Group",
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
    line_kwgs: dict = None,
    alpha: float = 0.6,
    show_centroids: bool = False,
):
    """
    Plot residuals vs predicted values by group, with options for overlay,
    single, or subplot layout.

    Parameters
    ----------
    data : dict — Group-level data with 'y_true' and 'y_prob' (or 'y_actual' / 'y_pred').
    save_path : str, optional — Directory to save the plot; displays if None.
    filename : str — Base filename for the saved plot.
    title : str — Title for the plot or figure.
    figsize : tuple — Figure size as (width, height).
    dpi : int — Resolution of the figure in dots per inch.
    label_fontsize : int — Font size for axis labels and titles.
    tick_fontsize : int — Font size for tick marks and legend text.
    decimal_places : int — Decimal places to show in summary stats.
    subplots : bool — If True, create separate subplots per group.
    n_cols : int — Number of subplot columns (when subplots=True).
    n_rows : int, optional — Number of subplot rows; inferred if not set.
    group : str, optional — If provided, only plot that group.
    color_by_group : bool — Use unique colors per group if True; gray otherwise.
    line_kwgs : dict, optional — Style dictionary for horizontal reference line.
    alpha : float — Transparency level for scatter points.
    show_centroids : bool — Plot centroid markers for each group's residuals if True.

    Raises
    ------
    ValueError — If both `group` and `subplots=True` are specified.

    Returns
    -------
    None — Saves or displays the resulting residual plot(s).
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    groups = sorted(data.keys())
    line_kwgs = line_kwgs or {"linestyle": "--", "color": "gray", "linewidth": 1}

    palette = (
        sns.color_palette("tab10", len(groups))
        if color_by_group
        else ["gray"] * len(groups)
    )
    color_map = dict(zip(groups, palette))

    if group:
        if group not in data:
            print(f"[Warning] Group '{group}' not found.")
            return
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        group_data = data[group]
        y_true = (
            group_data["y_true"] if "y_true" in group_data else group_data["y_actual"]
        )
        y_prob = (
            group_data["y_prob"] if "y_prob" in group_data else group_data["y_pred"]
        )
        _plot_single_residual_ax(
            ax,
            y_true,
            y_prob,
            group,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            alpha=alpha,
            color=color_map[group],
            line_kwargs=line_kwgs,
            show_centroid=show_centroids,
        )
        ax.set_title(f"{title} ({group})", fontsize=label_fontsize)
        fig.tight_layout()
        save_or_show_plot(fig, save_path, f"{filename}_{group}")
        return

    if subplots:
        if n_rows is None:
            n_rows = int(np.ceil(len(groups) / n_cols))
        elif n_rows * n_cols < len(groups):
            print(
                f"[Warning] Grid size {n_rows}x{n_cols} only supports "
                f"{n_rows * n_cols} plots; showing first {n_rows * n_cols} of "
                f"{len(groups)} groups."
            )

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
            dpi=dpi,
        )
        axes = axes.flatten()

        for i, grp in enumerate(groups):
            if i >= len(axes):
                break
            group_data = data[grp]
            yt = (
                group_data["y_true"]
                if "y_true" in group_data
                else group_data["y_actual"]
            )
            yp = (
                group_data["y_prob"] if "y_prob" in group_data else group_data["y_pred"]
            )
            _plot_single_residual_ax(
                axes[i],
                yt,
                yp,
                grp,
                label_fontsize=tick_fontsize + 2,
                tick_fontsize=tick_fontsize,
                alpha=alpha,
                color=color_map[grp],
                line_kwargs=line_kwgs,
                show_centroid=show_centroids,
            )

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=tick_fontsize + 3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_or_show_plot(fig, save_path, filename)
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    legend_entries = []

    for grp in groups:
        group_data = data[grp]
        yt = group_data["y_true"] if "y_true" in group_data else group_data["y_actual"]
        yp = group_data["y_prob"] if "y_prob" in group_data else group_data["y_pred"]
        residuals_grp = yt - yp

        _plot_single_residual_ax(
            ax,
            yt,
            yp,
            grp,
            label_fontsize=tick_fontsize + 2,
            tick_fontsize=tick_fontsize,
            alpha=alpha,
            color=color_map[grp],
            line_kwargs=line_kwgs,
            show_centroid=show_centroids,
        )

        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        residual_mean = np.mean(residuals_grp)
        n = len(yt)

        label = (
            f"R² for {grp} = {r2:.{decimal_places}f}, "
            f"MAE = {mae:.{decimal_places}f}, "
            f"Residual μ = {residual_mean:.{decimal_places}f}, "
            f"n = {n:,}"
        )

        legend_entries.append((label, color_map[grp]))

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
    ax.set_title(title, fontsize=tick_fontsize + 2)
    ax.set_xlabel("Predicted Value", fontsize=tick_fontsize)
    ax.set_ylabel("Residual (y_true - y_prob)", fontsize=tick_fontsize)
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show_plot(fig, save_path, f"{filename}_overlay")


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
    Helper function to plot the ROC curve for a single group on a given axis.

    Used by higher-level plotting routines to modularize group-wise plotting,
    apply consistent labeling, and support optional custom styling and legends.
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
    Plot ROC AUC curves by group with options for overlay, individual, or
    subplot layout.

    Parameters
    ----------
    data : dict — Dictionary of group-level data with 'y_true' and 'y_prob'.
    save_path : str, optional — Directory to save the plot; displays if None.
    filename : str — Base filename for the saved plot.
    title : str — Title for the plot or figure.
    figsize : tuple — Figure size as (width, height).
    dpi : int — Resolution of the figure in dots per inch.
    label_fontsize : int — Font size for axis labels and titles.
    tick_fontsize : int — Font size for tick labels and legend text.
    decimal_places : int — Number of decimal places in AUC and count stats.
    subplots : bool — If True, plots each group in a separate subplot.
    n_cols : int — Number of columns in subplot grid.
    n_rows : int, optional — Number of rows in subplot grid; inferred if not set.
    group : str, optional — If provided, plots only the specified group.
    color_by_group : bool — If True, assigns a unique color to each group.
    curve_kwgs : dict, optional — Per-group curve styling overrides.
    line_kwgs : dict, optional — Styling for the diagonal reference line.

    Raises
    ------
    ValueError — If both `group` and `subplots=True` are specified.

    Returns
    -------
    None — Saves or displays the resulting ROC curve plot(s).
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

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

        save_or_show_plot(fig, save_path, f"{filename}_overlay")


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
    Helper function to plot a Precision-Recall curve for a single group on the
    given axis.

    Used by grouped PR plotting routines to modularize per-group logic, apply
    consistent styling,
    and support optional customization of labels, legends, and baseline
    reference lines.
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
    Plot Precision-Recall curves by group with options for overlay, individual,
    or subplot layout.

    Parameters
    ----------
    data : dict — Dictionary of group-level data with 'y_true' and 'y_prob'.
    save_path : str, optional — Directory to save the plot; displays if None.
    filename : str — Base filename for the saved plot.
    title : str — Title for the plot or figure.
    figsize : tuple — Figure size as (width, height).
    dpi : int — Resolution of the figure in dots per inch.
    label_fontsize : int — Font size for axis labels and titles.
    tick_fontsize : int — Font size for tick labels and legend text.
    decimal_places : int — Number of decimal places in AP and count stats.
    subplots : bool — If True, plots each group in a separate subplot.
    n_cols : int — Number of columns in subplot grid.
    n_rows : int, optional — Number of rows in subplot grid; inferred if not set.
    group : str, optional — If provided, plots only the specified group.
    color_by_group : bool — If True, assigns a unique color to each group.
    curve_kwgs : dict, optional — Per-group curve styling overrides.
    line_kwgs : dict, optional — Styling for the baseline reference line.

    Raises
    ------
    ValueError — If both `group` and `subplots=True` are specified.

    Returns
    -------
    None — Saves or displays the resulting PR curve plot(s).
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    valid_data = {g: v for g, v in data.items() if len(set(v["y_true"])) > 1}
    groups = sorted(valid_data.keys())

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

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=figsize,
            dpi=dpi,
        )
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

        save_or_show_plot(fig, save_path, f"{filename}_overlay")


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
    """
    Helper function to plot a calibration curve for a single group on the given axis.

    Used by group-level calibration routines to modularize plotting logic,
    include Brier score summaries, and apply consistent styling, legends, and
    reference line.
    """

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
    Plot calibration curves by group with options for overlay, individual, or
    subplot layout.

    Parameters
    ----------
    data : dict — Dictionary of group-level data with 'y_true' and 'y_prob'.
    n_bins : int — Number of bins to compute calibration curve (default: 10).
    save_path : str, optional — Directory to save the plot; displays if None.
    filename : str — Base filename for the saved plot.
    title : str — Title for the plot or figure.
    figsize : tuple — Figure size as (width, height).
    dpi : int — Resolution of the figure in dots per inch.
    label_fontsize : int — Font size for axis labels and titles.
    tick_fontsize : int — Font size for tick labels and legend text.
    decimal_places : int — Decimal places for Brier score and axis values.
    subplots : bool — If True, plots each group in a separate subplot.
    n_cols : int — Number of columns in subplot grid.
    n_rows : int, optional — Number of rows in subplot grid; inferred if not set.
    group : str, optional — If provided, plots only the specified group.
    color_by_group : bool — If True, assigns a unique color to each group.
    curve_kwgs : dict, optional — Per-group curve styling overrides.
    line_kwgs : dict, optional — Styling for the diagonal reference line.

    Raises
    ------
    ValueError — If both `group` and `subplots=True` are specified.

    Returns
    -------
    None — Saves or displays the resulting calibration plot(s).
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

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

        save_or_show_plot(fig, save_path, f"{filename}_overlay")


################################################################################
# Disparity Metrics (Violin or Box Plots)
################################################################################


def get_layout(n_metrics, max_cols=None, figsize=None, strict_layout=True):
    """
    Determine subplot layout (rows, columns, figure size) for plotting multiple
    metrics.

    Supports both strict and flexible layout modes, with optional control over
    column count and figure dimensions to ensure clean, readable visualizations.
    """

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
    """
    Plot disparity metrics across categories using violin, box, or similar
    seaborn-based plots.

    Parameters
    ----------
    dispa : list — List of dictionaries containing group-wise metric values.
    metric_cols : list — Names of the metrics to visualize (e.g., TPR disparity,
    FPR disparity).
    name : str — Prefix used for subplot titles.
    plot_kind : str — Seaborn plot type to use (e.g., 'violinplot', 'boxplot').
    categories : list or str — List of category keys to include; use "all" for all keys.
    include_legend : bool — Whether to display a color-coded legend.
    cmap : str — Matplotlib colormap name for group coloring (default: "tab20c").
    save_path : str, optional — Directory to save the resulting plot; displays if None.
    filename : str — Base filename for the saved plot.
    max_cols : int, optional — Maximum number of subplot columns.
    strict_layout : bool — If True, enforces fixed figure height per row.
    figsize : tuple, optional — Custom figure size; overrides automatic layout.
    **plot_kwargs : dict — Additional keyword arguments passed to the seaborn plot
    function.

    Raises
    ------
    TypeError — If `dispa` is not a list.
    ValueError — If `plot_kind` is not a valid seaborn plot type.

    Returns
    -------
    None — Saves or displays the resulting disparity metrics plot.
    """

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

    save_or_show_plot(fig, save_path, f"{filename}_overlay")


################################################################################
# Bootstrapped ROC AUC and Precision-Recall Helper
################################################################################


def _plot_single_bootstrapped_curve_ax(
    ax,
    y_array,
    common_grid,
    group,
    label_prefix="AUCPR",
    label_fontsize=12,
    tick_fontsize=10,
    bar_every=10,
    curve_kwargs=None,
    fill_kwargs=None,
    line_kwargs=None,
    show_grid=True,
    x_label="Recall",
    y_label="Precision",
    show_reference_line=False,
):
    """
    Helper function to plot a bootstrapped performance curve with confidence
    bands for a single group.

    Used to visualize variability across bootstrapped iterations, including
    error bars, shaded confidence intervals, and optional reference lines
    (e.g., for AUROC).
    """

    mean_y = np.mean(y_array, axis=0)
    lower = np.percentile(y_array, 2.5, axis=0)
    upper = np.percentile(y_array, 97.5, axis=0)
    aucs = [np.trapz(y, common_grid) for y in y_array]
    mean_auc = np.mean(aucs)
    lower_auc = np.percentile(aucs, 2.5)
    upper_auc = np.percentile(aucs, 97.5)
    label = f"{group} (Mean {label_prefix} = {mean_auc:.2f} [{lower_auc:.2f}, {upper_auc:.2f}])"

    curve_kwargs = curve_kwargs or {}
    fill_kwargs = fill_kwargs or {
        "alpha": 0.2,
        "color": curve_kwargs.get("color", "#1f77b4"),
    }
    line_kwargs = line_kwargs or {"color": "gray", "linestyle": "--", "linewidth": 1}

    ax.plot(common_grid, mean_y, label=label, **curve_kwargs)
    ax.fill_between(common_grid, lower, upper, **fill_kwargs)

    for j in range(0, len(common_grid), int(np.ceil(len(common_grid) / bar_every))):
        x_val = common_grid[j]
        mean_val = mean_y[j]
        err_low = mean_val - lower[j]
        err_high = upper[j] - mean_val
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

    if show_reference_line and label_prefix == "AUROC":
        ax.plot([0, 1], [0, 1], **line_kwargs)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(group, fontsize=label_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(loc="lower right", fontsize=tick_fontsize)
    if show_grid:
        ax.grid(True)


################################################################################
# Bootstrapped ROC AUC Curve Plot
################################################################################


def eq_plot_bootstrapped_roc_curves(
    boot_sliced_data,
    title="Bootstrapped ROC Curves by Group",
    filename="roc_curves_by_group",
    save_path=None,
    dpi=100,
    figsize=(6, 5),
    common_grid=np.linspace(0, 1, 100),
    bar_every=10,
    label_fontsize=12,
    tick_fontsize=10,
    curve_kwgs=None,
    fill_kwgs=None,
    line_kwgs=None,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    show_grid=False,
    color_by_group=True,
    uniform_color: str = None,
):
    """
    Plot bootstrapped ROC curves by group, with support for overlay, subplots,
    or single-group display.

    Parameters
    ----------
    boot_sliced_data : list — List of dictionaries with 'y_true' and 'y_prob'
    per group per bootstrap iteration.
    title : str — Title for the plot or figure.
    filename : str — Base filename for the saved plot.
    save_path : str, optional — Directory to save the resulting plot; displays if None.
    dpi : int — Resolution of the figure in dots per inch.
    figsize : tuple — Base figure size for each subplot (width, height).
    common_grid : np.ndarray — X-axis values to interpolate across (typically 0 to 1).
    bar_every : int — Frequency of vertical error bars along the curve.
    label_fontsize : int — Font size for axis labels and titles.
    tick_fontsize : int — Font size for tick labels and legend text.
    curve_kwgs : dict, optional — Per-group overrides for the line style of the
    mean curve.
    fill_kwgs : dict, optional — Per-group overrides for confidence interval shading.
    line_kwgs : dict, optional — Style for the reference line (diagonal).
    subplots : bool — If True, creates subplots per group; otherwise overlays
    all curves.
    n_cols : int — Number of subplot columns when subplots=True.
    n_rows : int, optional — Number of subplot rows; inferred if not set.
    group : str, optional — If provided, plots only the specified group.
    show_grid : bool — Whether to enable grid on each subplot.
    color_by_group : bool — If True, assigns unique colors per group.
    uniform_color : str, optional — If provided, applies this color to all
    groups (overrides palette).

    Raises
    ------
    ValueError — If both `group` and `subplots=True` are specified.

    Returns
    -------
    None — Saves or displays the resulting ROC curve plot(s).
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    group_fpr_tpr = {}

    for bootstrap_iter in boot_sliced_data:
        for grp, values in bootstrap_iter.items():
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

            if grp not in group_fpr_tpr:
                group_fpr_tpr[grp] = []

            group_fpr_tpr[grp].append(tpr_interp)

    group_names = sorted(group_fpr_tpr.keys())

    if color_by_group:
        palette = plt.get_cmap("tab10").colors
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(group_names)}
    else:
        fallback_color = uniform_color or "#1f77b4"
        color_map = {g: fallback_color for g in group_names}

    def get_plot_kwargs(grp):
        group_curve_kwgs = {"color": color_map[grp]}
        if curve_kwgs and grp in curve_kwgs:
            group_curve_kwgs.update(curve_kwgs[grp])

        group_fill_kwgs = {"alpha": 0.2, "color": color_map[grp]}
        if fill_kwgs and grp in fill_kwgs:
            group_fill_kwgs.update(fill_kwgs[grp])

        return group_curve_kwgs, group_fill_kwgs

    if group:
        if group not in group_fpr_tpr:
            print(f"[Warning] Group '{group}' not found in bootstrapped data.")
            return

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        y_array = np.vstack([y for y in group_fpr_tpr[group] if not np.isnan(y).any()])
        group_curve_kwgs, group_fill_kwgs = get_plot_kwargs(group)

        _plot_single_bootstrapped_curve_ax(
            ax=ax,
            y_array=y_array,
            common_grid=common_grid,
            group=group,
            label_prefix="AUROC",
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            bar_every=bar_every,
            curve_kwargs=group_curve_kwgs,
            fill_kwargs=group_fill_kwgs,
            line_kwargs=line_kwgs,
            show_grid=show_grid,
            show_reference_line=True,
            x_label="False Positive Rate",
            y_label="True Positive Rate",
        )

        fig.suptitle(f"{title} ({group})", fontsize=label_fontsize + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    elif subplots:
        num_groups = len(group_names)
        if n_rows is None:
            n_rows = int(np.ceil(num_groups / n_cols))
        elif n_rows * n_cols < num_groups:
            print(
                f"[Warning] Grid size {n_rows}x{n_cols} only supports {n_rows * n_cols} plots; "
                f"showing first {n_rows * n_cols} of {num_groups} groups."
            )

        fig_w = figsize[0] * n_cols
        fig_h = figsize[1] * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
        axes = axes.flatten()

        for i, grp in enumerate(group_names):
            if i >= len(axes):
                break

            ax = axes[i]
            y_array = np.vstack(
                [y for y in group_fpr_tpr[grp] if not np.isnan(y).any()]
            )
            group_curve_kwgs, group_fill_kwgs = get_plot_kwargs(grp)

            _plot_single_bootstrapped_curve_ax(
                ax=ax,
                y_array=y_array,
                common_grid=common_grid,
                group=grp,
                label_prefix="AUROC",
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                bar_every=bar_every,
                curve_kwargs=group_curve_kwgs,
                fill_kwargs=group_fill_kwgs,
                line_kwargs=line_kwgs,
                show_grid=show_grid,
                show_reference_line=True,
                x_label="False Positive Rate",
                y_label="True Positive Rate",
            )

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=label_fontsize + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for grp in group_names:
            y_array = np.vstack(
                [y for y in group_fpr_tpr[grp] if not np.isnan(y).any()]
            )
            if y_array.shape[0] == 0:
                continue

            group_curve_kwgs, group_fill_kwgs = get_plot_kwargs(grp)

            _plot_single_bootstrapped_curve_ax(
                ax=ax,
                y_array=y_array,
                common_grid=common_grid,
                group=grp,
                label_prefix="AUROC",
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                bar_every=bar_every,
                curve_kwargs=group_curve_kwgs,
                fill_kwargs=group_fill_kwgs,
                line_kwargs=line_kwgs,
                show_grid=show_grid,
                show_reference_line=True,
                x_label="False Positive Rate",
                y_label="True Positive Rate",
            )

        ax.set_title(title, fontsize=label_fontsize + 2)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fontsize=tick_fontsize,
            ncol=1,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_or_show_plot(fig, save_path, f"{filename}_overlay")


################################################################################
# Bootstrapped Precision-Recall Curve Plot
################################################################################


def eq_plot_bootstrapped_pr_curves(
    boot_sliced_data,
    title="Bootstrapped PR Curves by Group",
    filename="pr_curves_by_group",
    save_path=None,
    dpi=100,
    figsize=(8, 6),
    common_grid=np.linspace(0, 1, 100),
    bar_every=10,
    label_fontsize=12,
    tick_fontsize=10,
    curve_kwgs=None,
    fill_kwgs=None,
    line_kwgs=None,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    show_grid=True,
    color_by_group=True,
    uniform_color: str = None,
):
    """
    Plot bootstrapped Precision-Recall (PR) curves by group with overlay,
    subplots, or single-group view.

    Parameters
    ----------
    boot_sliced_data : list — List of dicts with 'y_true' and 'y_prob' per group
    per bootstrap iteration.
    title : str — Title of the plot or figure.
    filename : str — Base filename to use when saving the plot.
    save_path : str, optional — Directory path to save the plot; shows plot if None.
    dpi : int — Dots per inch for figure resolution.
    figsize : tuple — Base size for each subplot or overlay figure.
    common_grid : np.ndarray — Grid of recall values to interpolate PR curves on.
    bar_every : int — Controls frequency of error bars plotted on the curve.
    label_fontsize : int — Font size for titles and axis labels.
    tick_fontsize : int — Font size for tick marks and legend text.
    curve_kwgs : dict, optional — Per-group line styling for the mean PR curve.
    fill_kwgs : dict, optional — Per-group fill styling for confidence intervals.
    line_kwgs : dict, optional — Line style for any reference lines.
    subplots : bool — Whether to display PR curves in subplots per group.
    n_cols : int — Number of columns when using subplots.
    n_rows : int, optional — Number of subplot rows; inferred if None.
    group : str, optional — If specified, plots only this group.
    show_grid : bool — Whether to enable grid lines in each subplot.
    color_by_group : bool — If True, assigns a distinct color per group.
    uniform_color : str, optional — Use the same color across groups, overrides
    palette.

    Raises
    ------
    ValueError — If both `group` and `subplots=True` are set simultaneously.

    Returns
    -------
    None — Displays or saves the bootstrapped PR curve visualization.
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    group_precision = {}

    for bootstrap_iter in boot_sliced_data:
        for grp, values in bootstrap_iter.items():
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
                precision_interp = interp(common_grid)
            except ValueError:
                precision_interp = np.full_like(common_grid, np.nan)

            if grp not in group_precision:
                group_precision[grp] = []

            group_precision[grp].append(precision_interp)

    group_names = sorted(group_precision.keys())

    if color_by_group:
        palette = plt.get_cmap("tab10").colors
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(group_names)}
    else:
        fallback_color = uniform_color or "#1f77b4"
        color_map = {g: fallback_color for g in group_names}

    def get_plot_kwargs(grp):
        group_curve_kwgs = {"color": color_map[grp]}
        if curve_kwgs and grp in curve_kwgs:
            group_curve_kwgs.update(curve_kwgs[grp])

        group_fill_kwgs = {"alpha": 0.2, "color": color_map[grp]}
        if fill_kwgs and grp in fill_kwgs:
            group_fill_kwgs.update(fill_kwgs[grp])

        return group_curve_kwgs, group_fill_kwgs

    if group:
        if group not in group_precision:
            print(f"[Warning] Group '{group}' not found in bootstrapped data.")
            return

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        y_array = np.vstack(
            [y for y in group_precision[group] if not np.isnan(y).any()]
        )
        group_curve_kwgs, group_fill_kwgs = get_plot_kwargs(group)

        _plot_single_bootstrapped_curve_ax(
            ax=ax,
            y_array=y_array,
            common_grid=common_grid,
            group=group,
            label_prefix="AUCPR",
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            bar_every=bar_every,
            curve_kwargs=group_curve_kwgs,
            fill_kwargs=group_fill_kwgs,
            line_kwargs=line_kwgs,
            show_grid=show_grid,
            show_reference_line=False,
            x_label="Recall",
            y_label="Precision",
        )

        fig.suptitle(f"{title} ({group})", fontsize=label_fontsize + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_or_show_plot(fig, save_path=save_path, filename=f"{filename}_{group}")

    elif subplots:
        num_groups = len(group_names)
        if n_rows is None:
            n_rows = int(np.ceil(num_groups / n_cols))
        elif n_rows * n_cols < num_groups:
            print(
                f"[Warning] Grid size {n_rows}x{n_cols} only supports {n_rows * n_cols} plots; "
                f"showing first {n_rows * n_cols} of {num_groups} groups."
            )

        fig_w = figsize[0] * n_cols
        fig_h = figsize[1] * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
        axes = axes.flatten()

        for i, grp in enumerate(group_names):
            if i >= len(axes):
                break

            ax = axes[i]
            y_array = np.vstack(
                [y for y in group_precision[grp] if not np.isnan(y).any()]
            )
            group_curve_kwgs, group_fill_kwgs = get_plot_kwargs(grp)

            _plot_single_bootstrapped_curve_ax(
                ax=ax,
                y_array=y_array,
                common_grid=common_grid,
                group=grp,
                label_prefix="AUCPR",
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                bar_every=bar_every,
                curve_kwargs=group_curve_kwgs,
                fill_kwargs=group_fill_kwgs,
                line_kwargs=line_kwgs,
                show_grid=show_grid,
                show_reference_line=False,
                x_label="Recall",
                y_label="Precision",
            )

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=label_fontsize + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_or_show_plot(fig, save_path=save_path, filename=filename)

    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for grp in group_names:
            y_array = np.vstack(
                [y for y in group_precision[grp] if not np.isnan(y).any()]
            )
            if y_array.shape[0] == 0:
                continue

            group_curve_kwgs, group_fill_kwgs = get_plot_kwargs(grp)

            _plot_single_bootstrapped_curve_ax(
                ax=ax,
                y_array=y_array,
                common_grid=common_grid,
                group=grp,
                label_prefix="AUCPR",
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
                bar_every=bar_every,
                curve_kwargs=group_curve_kwgs,
                fill_kwargs=group_fill_kwgs,
                line_kwargs=line_kwgs,
                show_grid=show_grid,
                show_reference_line=False,
                x_label="Recall",
                y_label="Precision",
            )

        ax.set_title(title, fontsize=label_fontsize + 2)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fontsize=tick_fontsize,
            ncol=1,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_or_show_plot(fig, save_path, f"{filename}_overlay")


################################################################################
# Bootstrapped Calibration Curve Plot
################################################################################


def eq_plot_bootstrapped_calibration_curves(
    boot_sliced_data,
    title="Bootstrapped Calibration Curves by Group",
    filename="calibration_curves_by_group",
    save_path=None,
    dpi=100,
    figsize=(8, 6),
    figsize_per_plot=(6, 5),
    n_bins=10,
    bar_every=10,
    label_fontsize=12,
    tick_fontsize=10,
    alpha_fill=0.2,
    curve_kwgs=None,
    fill_kwgs=None,
    line_kwgs=None,
    subplots=False,
    n_cols=2,
    n_rows=None,
    group=None,
    color_by_group=True,
    uniform_color="#1f77b4",
    decimal_places=2,
):
    """
    Plot bootstrapped calibration curves by group with support for overlay,
    subplots, or single-group view.

    Parameters
    ----------
    boot_sliced_data : list — List of dicts per bootstrap iteration with 'y_true'
    and 'y_prob' keys for each group.
    title : str — Title for the overall plot or subplot grid.
    filename : str — Base filename to use when saving the figure.
    save_path : str, optional — Directory to save the plot; if None, the figure
    is displayed.
    dpi : int — Figure resolution in dots per inch.
    figsize : tuple — Size of the overall figure (used in overlay or single-group modes).
    figsize_per_plot : tuple — Size per subplot when using subplots.
    n_bins : int — Number of bins used for calibration binning.
    bar_every : int — Frequency of error bars shown on calibration curve points.
    label_fontsize : int — Font size for axis labels and titles.
    tick_fontsize : int — Font size for tick marks and legend text.
    alpha_fill : float — Opacity for the confidence interval band.
    curve_kwgs : dict, optional — Per-group styling for the mean calibration line.
    fill_kwgs : dict, optional — Per-group styling for the confidence band.
    line_kwgs : dict, optional — Per-group styling for the diagonal reference line.
    subplots : bool — Whether to generate subplot layout by group.
    n_cols : int — Number of columns in the subplot grid.
    n_rows : int, optional — Number of rows in the subplot grid; inferred if None.
    group : str, optional — If specified, plots calibration for this group only.
    color_by_group : bool — If True, assigns distinct colors to each group.
    uniform_color : str — Fallback color to use when color_by_group is False.
    decimal_places : int — Number of decimals to round metrics in plot labels.

    Raises
    ------
    ValueError — If both `group` and `subplots=True` are passed simultaneously.

    Returns
    -------
    None — Displays or saves the bootstrapped calibration curve visualization.
    """

    if group is not None and subplots:
        raise ValueError("Cannot use subplots=True when a specific group is selected.")

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    group_cal, group_brier = {}, {}

    for bootstrap_iter in boot_sliced_data:
        for grp, values in bootstrap_iter.items():
            y_true, y_prob = values["y_true"], values["y_prob"]

            frac_positives = np.full(n_bins, np.nan)
            for i in range(n_bins):
                mask = (y_prob >= bins[i]) & (
                    y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1]
                )
                if np.any(mask):
                    frac_positives[i] = np.mean(y_true[mask])

            group_cal.setdefault(grp, []).append(frac_positives)
            group_brier.setdefault(grp, []).append(brier_score_loss(y_true, y_prob))

    group_names = sorted(group_cal.keys())
    if color_by_group:
        palette = plt.get_cmap("tab10").colors
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(group_names)}
    else:
        color_map = {g: uniform_color for g in group_names}

    def get_plot_kwargs(grp):
        group_curve_kwgs = {"color": color_map[grp]}
        if curve_kwgs and grp in curve_kwgs:
            group_curve_kwgs.update(curve_kwgs[grp])
        group_fill_kwgs = {"alpha": alpha_fill, "color": color_map[grp]}
        if fill_kwgs and grp in fill_kwgs:
            group_fill_kwgs.update(fill_kwgs[grp])
        group_line_kwgs = {"color": "gray", "linestyle": "--", "linewidth": 1}
        if line_kwgs and grp in line_kwgs:
            group_line_kwgs.update(line_kwgs[grp])
        return group_curve_kwgs, group_fill_kwgs, group_line_kwgs

    def plot_group(ax, grp):
        cal_array = np.array(group_cal[grp])
        valid = ~np.all(np.isnan(cal_array), axis=1)
        cal_array = cal_array[valid]
        if cal_array.shape[0] == 0:
            ax.set_title(grp, fontsize=label_fontsize)
            ax.plot([0, 1], [0, 1], "--", color="gray")
            return

        mean, lower, upper = (
            np.nanmean(cal_array, axis=0),
            np.nanpercentile(cal_array, 2.5, axis=0),
            np.nanpercentile(cal_array, 97.5, axis=0),
        )
        briers = np.array(group_brier[grp])
        mb, lb, ub = (
            np.mean(briers),
            np.percentile(briers, 2.5),
            np.percentile(briers, 97.5),
        )
        label = f"{grp} (Mean Brier = {mb:.{decimal_places}f} "
        f"[{lb:.{decimal_places}f}, {ub:.{decimal_places}f}])"

        curve_kw, fill_kw, line_kw = get_plot_kwargs(grp)
        ax.plot(bin_centers, mean, label=label, marker="o", **curve_kw)
        ax.fill_between(bin_centers, lower, upper, **fill_kw)

        for j in np.linspace(0, n_bins - 1, bar_every, dtype=int):
            ax.errorbar(
                bin_centers[j],
                mean[j],
                yerr=[[mean[j] - lower[j]], [upper[j] - mean[j]]],
                fmt="o",
                color=curve_kw["color"],
                markersize=3,
                capsize=2,
                elinewidth=1,
                alpha=0.6,
            )

        ax.plot([0, 1], [0, 1], **line_kw)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(grp, fontsize=label_fontsize)
        ax.set_xlabel("Predicted Probability", fontsize=label_fontsize)
        ax.set_ylabel("Fraction of Positives", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.legend(loc="lower right", fontsize=tick_fontsize)
        ax.grid(True)

    if group:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plot_group(ax, group)
        fig.suptitle(f"{title} ({group})", fontsize=label_fontsize + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_or_show_plot(fig, save_path, f"{filename}_{group}")

    elif subplots:
        num_groups = len(group_names)
        if n_rows is None:
            n_rows = int(np.ceil(num_groups / n_cols))
        elif n_rows * n_cols < num_groups:
            print(
                f"[Warning] Grid size {n_rows}x{n_cols} only supports "
                f"{n_rows * n_cols} plots; showing first {n_rows * n_cols} of "
                f"{num_groups} groups."
            )

        fig_w = figsize_per_plot[0] * n_cols
        fig_h = figsize_per_plot[1] * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
        axes = axes.flatten()

        for i, grp in enumerate(group_names):
            if i >= len(axes):
                break
            plot_group(axes[i], grp)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=label_fontsize + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_or_show_plot(fig, save_path, filename)

    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for grp in group_names:
            plot_group(ax, grp)

        ax.set_title(title, fontsize=label_fontsize + 2)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fontsize=tick_fontsize,
            ncol=1,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_or_show_plot(fig, save_path, f"{filename}_overlay")


################################################################################
# Grup Metrics Extraction and Confidence Intervals
################################################################################
def extract_group_metrics(race_metrics):
    """
    Extract TPR and FPR values by group from a list of bootstrapped metric
    dictionaries.

    Returns a dictionary of lists for each group's metrics and the set of unique
    group names, used to support group-wise performance analysis and visualization.
    """

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
    """
    Compute confidence intervals for group-level metrics across bootstrapped
    samples.

    Takes a dictionary of metric lists and returns lower and upper bounds for
    each metric based on the specified confidence level (default is 95%).
    """

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
