import numpy as np
import pandas as pd
import equiboots as eqb


if __name__ == "__main__":
    # Generate synthetic test data
    y_prob = np.random.rand(1000)
    y_pred = y_prob > 0.5
    y_true = np.random.randint(0, 2, 1000)

    race = (
        np.random.RandomState(3)
        .choice(["white", "black", "asian", "hispanic"], 1000)
        .reshape(-1, 1)
    )
    sex = np.random.choice(["M", "F"], 1000).reshape(-1, 1)

    fairness_df = pd.DataFrame(
        data=np.concatenate((race, sex), axis=1), columns=["race", "sex"]
    )

    # Initialize and process groups
    eq = eqb.EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
    )
    eq.grouper(groupings_vars=["race", "sex"])
    sliced_data = eq.slicer("race")

    eqb.eq_plot_group_curves(
        sliced_data,
        curve_type="pr",
        title="ROC AUC by Race Group",
        n_bins=10,
    )

    eqb.eq_plot_group_curves(
        sliced_data,
        curve_type="roc",
        title="ROC AUC by Race Group",
        n_bins=10,
    )

    # Regular calibration ex. w/ calibration area printed to legend
    eqb.eq_plot_group_curves(
        sliced_data,
        curve_type="calibration",
        title="ROC AUC by Race Group",
        n_bins=10,
    )

    # LOWESS smoothing calibration
    eqb.eq_plot_group_curves(
        sliced_data,
        curve_type="calibration",
        title="ROC AUC by Race Group",
        n_bins=10,
        lowess=0.5,
    )
