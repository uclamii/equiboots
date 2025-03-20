from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from EquiBoots import EquiBoots
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
    dpi: int = 300,
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
        plt.close(fig)

    return fig


################################################################################
# Precision-Recall Curve Plot
################################################################################


def eq_plot_precision_recall(
    data: dict,
    save_path: str = None,
    filename: str = "precision_recall_by_group",
    title: str = "Precision-Recall Curve by Group",
    figsize: tuple = (8, 6),
    dpi: int = 300,
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
        plt.close(fig)

    return fig


################################################################################
# Calibration Curve Plot
################################################################################


def eq_calibration_curve_plot(
    data: dict,
    n_bins: int = 10,
    figsize: tuple = (8, 6),
    dpi: int = 300,
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
        plt.close(fig)

    return fig


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
    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
    )
    eq.grouper(groupings_vars=["race", "sex"])
    sliced_data = eq.slicer("race")

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
