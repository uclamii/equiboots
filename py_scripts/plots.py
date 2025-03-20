from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from EquiBoots import EquiBoots
from sklearn.metrics import roc_curve, auc

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

  
# calibration curve plot
def eq_calibration_curve_plot(data, n_bins=10, ax=None):
    """Plot calibration curve for binary classification."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for key in data.keys():
        y_true = data[key]["y_true"]
        y_prob = data[key]["y_prob"]
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)
        # Plot each calibration curve on the same axis
        ax.plot(mean_predicted_value, fraction_of_positives, marker='o', label=key)

    # Add the perfectly calibrated line
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')
    ax.set_xlabel('Mean predicted value')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration plot')
    ax.legend()
    return ax


if __name__ == "__main__":

    # Generate fake data
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

    eq = EquiBoots(y_true, y_prob, y_pred, fairness_df, fairness_vars=["race", "sex"])

    eq.grouper(groupings_vars=["race", "sex"])

    data = eq.slicer("race")
    # plt.show()
    
    # Generate the calibration plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    eq_calibration_curve_plot(data, n_bins=10, ax=ax)
    plt.savefig("plots/calibration_plot.png")  # Save the plot as a PNG file
    np.hstack((race, sex)),
    columns=["race", "sex"],
    )

    # Initialize and run EquiBoots
    eq = EquiBoots(
        y_true,
        y_prob,
        y_pred,
        fairness_df,
        fairness_vars=["race", "sex"],
    )
    eq.grouper(["race"])
    sliced_data = eq.slicer("race")

    # Call the plot function
    fig = eq_plot_roc_auc(
        data=sliced_data,
        save_path=None,  # Or e.g., "./figures"
        filename="roc_by_race",
        title="ROC Curve by Race",
        tick_fontsize=8,
        decimal_places=3,
    )

    fig.show()  # Show the plot during testing

