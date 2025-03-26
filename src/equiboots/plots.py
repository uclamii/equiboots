from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
import os
import math

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

################################################################################
# ROC AUC Curve Plot
################################################################################


def eq_plot_roc_auc(
    data: dict,
    save_path: str = None,
    filename: str = "roc_auc_by_group",
    title: str = "ROC Curve by Group",
    figsize: tuple = (8, 6),
    dpi: int = 100,
    tick_fontsize: int = 10,
    decimal_places: int = 2,
):
    """
    Plots ROC AUC curves for each group in a fairness dictionary.

    Parameters
    ----------
    data : dict
        Dictionary with group names as keys and 'y_true' and 'y_prob' arrays as values.
    save_path : str, optional
        Directory to save the plot. If None, the plot is returned.
    filename : str, optional
        Name of the output file (no extension).
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Size of the plot.
    dpi : int, optional
        Resolution of the plot.
    tick_fontsize : int, optional
        Font size for legend text.
    decimal_places : int, optional
        Decimal precision for AUC in the legend.

    Returns
    -------
    matplotlib.figure.Figure
        The ROC AUC plot figure.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for group, values in sorted(data.items()):
        y_true = values["y_true"]
        y_prob = values["y_prob"]

        if len(set(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        total = len(y_true)
        positives = np.sum(y_true)
        negatives = total - positives

        label = (
            f"AUC for {group} = {roc_auc:.{decimal_places}f}, "
            f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,}"
        )
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fontsize=tick_fontsize,
        ncol=1,
    )
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)  # prevent display in notebook
    else:
        plt.show()  # only show if not saving


################################################################################
# Precision-Recall Curve Plot
################################################################################


def eq_plot_precision_recall(
    data: dict,
    save_path: str = None,
    filename: str = "precision_recall_by_group",
    title: str = "Precision-Recall Curve by Group",
    figsize: tuple = (8, 6),
    dpi: int = 100,
    tick_fontsize: int = 10,
    decimal_places: int = 2,
):
    """
    Plot Precision-Recall curves for each group in a fairness category.

    Parameters:
    -----------
    data : dict
        Dictionary of the form {group_name: {'y_true': ..., 'y_prob': ...}}
    save_path : str or None
        If specified, the figure is saved to this path with the given filename.
    filename : str
        Filename for saving the figure (without extension).
    title : str
        Title for the plot.
    figsize : tuple
        Size of the figure (width, height).
    dpi : int
        Dots per inch for the saved figure.
    tick_fontsize : int
        Font size for legend labels.
    decimal_places : int
        Number of decimal places to display for average precision scores.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for group, values in sorted(data.items()):
        y_true = values["y_true"]
        y_prob = values["y_prob"]

        if len(set(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        total = len(y_true)
        positives = int(np.sum(y_true))
        negatives = total - positives

        label = (
            f"AP for {group} = {avg_precision:.{decimal_places}f}, "
            f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,}"
        )
        ax.plot(recall, precision, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fontsize=tick_fontsize,
        ncol=1,
    )
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)  # prevent display in notebook
    else:
        plt.show()  # only show if not saving


################################################################################
# Calibration Curve Plot
################################################################################


def eq_calibration_curve_plot(
    data: dict,
    n_bins: int = 10,
    figsize: tuple = (8, 6),
    dpi: int = 100,
    title: str = "Calibration Curve by Group",
    filename: str = "calibration_by_group",
    save_path: str = None,
    ax=None,
    tick_fontsize: int = 10,
    decimal_places: int = 2,
):
    """
    Plot calibration curves for each group in a fairness category.

    Parameters:
    -----------
    data : dict
        Dictionary of the form {group_name: {'y_true': ..., 'y_prob': ...}}.
    n_bins : int
        Number of bins to use for calibration curve.
    figsize : tuple
        Size of the figure.
    dpi : int
        Dots per inch (resolution).
    title : str
        Title of the plot.
    filename : str
        Filename to use when saving the plot (without extension).
    save_path : str or None
        Directory to save the plot. If None, the figure is returned instead.
    ax : matplotlib.axes._axes.Axes or None
        Axis to plot on. If None, a new figure and axis are created.
    tick_fontsize : int
        Font size for axis labels and legend.
    decimal_places : int
        Number of decimal places to show for Brier scores.
    Returns:
    --------
    matplotlib.figure.Figure
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    for key in sorted(data.keys()):
        y_true = data[key]["y_true"]
        y_prob = data[key]["y_prob"]

        # Compute calibration curve and Brier score
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true,
            y_prob,
            n_bins=n_bins,
        )
        brier = brier_score_loss(y_true, y_prob)
        total = len(y_true)
        positives = int(np.sum(y_true))
        negatives = total - positives

        label = (
            f"{key} (Brier = {brier:.{decimal_places}f}, "
            f"Count: {total:,}, Pos: {positives:,}, Neg: {negatives:,})"
        )

        ax.plot(
            np.round(mean_predicted_value, decimal_places),
            np.round(fraction_of_positives, decimal_places),
            marker="o",
            label=label,
        )

    # Perfect calibration line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfectly calibrated",
        color="gray",
    )

    ax.set_xlabel("Mean predicted value", fontsize=tick_fontsize)
    ax.set_ylabel("Fraction of positives", fontsize=tick_fontsize)
    ax.set_title(title, fontsize=tick_fontsize + 2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fontsize=tick_fontsize,
        ncol=1,
    )
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)  # prevent display in notebook
    else:
        plt.show()  # only show if not saving


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

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)  # prevent display in notebook
    else:
        plt.show()  # only show if not saving


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

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


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

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


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
    Plot bootstrapped calibration curves (fraction of positives vs. predicted probability)
    with shaded confidence intervals, one group per subplot (grid layout). The curves
    are computed using fixed bins so that they remain jagged (i.e., not smoothed).

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

        # Plot the mean calibration curve (using bin centers) with markers to show jagged steps
        ax.plot(bin_centers, mean_cal, label=brier_str, color=color, marker="o")

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

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"{filename}.png"), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


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
