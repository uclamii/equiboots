from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from EquiBoots import EquiBoots

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


if __name__ == "__main__":

    # Generate synthetic test data
    y_prob = np.random.rand(1000)
    y_pred = y_prob > 0.5
    y_true = np.random.randint(0, 2, 1000)

    race = np.random.choice(["white", "black", "asian", "hispanic"], 1000).reshape(
        -1, 1
    )
    sex = np.random.choice(["M", "F"], 1000).reshape(-1, 1)

    fairness_df = pd.DataFrame(
        data=np.concatenate((race, sex), axis=1), columns=["race", "sex"]
    )

    # Initialize and process groups
    eq1 = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
    )
    eq1.grouper(groupings_vars=["race", "sex"])
    sliced_data = eq1.slicer("race")

    eq2 = EquiBoots(
        y_true,
        y_prob,
        y_pred,
        fairness_df,
        fairness_vars=["race", "sex"],
        reference_groups=["white", "M"],
        task="binary_classification",
        bootstrap_flag=True,
        num_bootstraps=10,
        boot_sample_size=100,
        balanced=False,  # False is stratified, True is balanced
    )

    # Set seeds
    eq2.set_fix_seeds([42, 123, 222, 999])
    eq2.grouper(groupings_vars=["race", "sex"])
    data = eq2.slicer("race")
    race_metrics = eq2.get_metrics(data)
    dispa = eq2.calculate_disparities(race_metrics, "race")

    ############################################################################
    # Plots
    ############################################################################

    # ROC plot
    fig1 = eq_plot_roc_auc(
        data=sliced_data,
        title="ROC Curve by Race",
        tick_fontsize=8,
        decimal_places=3,
    )
    fig1.show()

    # Precision-Recall plot
    fig2 = eq_plot_precision_recall(
        data=sliced_data,
        title="Precision-Recall Curve by Race",
        tick_fontsize=8,
        decimal_places=3,
    )
    fig2.show()

    # Calibration plot
    fig3 = eq_calibration_curve_plot(
        data=sliced_data,
        n_bins=10,
        title="Calibration Curve by Race",
        tick_fontsize=8,
        decimal_places=3,
    )
    fig3.show()

    fig4 = eq_disparity_metrics_plot(
        dispa,
        metric_cols=["Accuracy_ratio", "Precision_ratio"],
        name="race",
        categories="all",
    )

    fig4.show()
