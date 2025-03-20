from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from EquiBoots import EquiBoots

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