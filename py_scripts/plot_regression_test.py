import numpy as np
import pandas as pd
import equiboots as eqb


if __name__ == "__main__":
    ## Generate synthetic regression-like data
    np.random.seed(42)
    y_true = np.random.normal(loc=50, scale=10, size=1000)  ## continuous target
    y_pred = y_true + np.random.normal(
        loc=0, scale=5, size=1000
    )  # predicted value with noise

    # Not really 'prob', but using this slot for predicted values
    y_prob = y_pred

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
    eq3 = eqb.EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        task="regression",
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
    )
    eq3.grouper(groupings_vars=["race", "sex"])
    sliced_data_2 = eq3.slicer("race")

    eqb.eq_plot_residuals_by_group(
        data=sliced_data_2,
        # y_true=y_true,
        # y_prob=y_pred,
        # group="black",
        title="Residuals by Race",
        filename="residuals_by_race",
        # subplots=True,
        # group="black",
        color_by_group=True,
        # n_cols=1,
        # n_rows=2,
        figsize=(8, 6),
        # group="black",
        show_centroids=True,
        # exclude_groups="white",
        show_grid=False,
    )
