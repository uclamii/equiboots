import numpy as np
import pandas as pd
import equiboots as eqb


if __name__ == "__main__":
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

    eq = eqb.EquiBoots(
        y_true,
        y_prob,
        y_pred,
        fairness_df,
        fairness_vars=["race", "sex"],
        reference_groups=["white", "M"],
        task="binary_classification",
        bootstrap_flag=False,
    )

    eq.grouper(groupings_vars=["race", "sex"])
    sliced_data = eq.slicer("race")

    data_race = eq.slicer("race")
    data_sex = eq.slicer("sex")

    race_metrics_3 = eq.get_metrics(data_race)
    sex_metrics_3 = eq.get_metrics(data_sex)

    dispa_race = eq.calculate_disparities(race_metrics_3, "race")
    dispa_sex = eq.calculate_disparities(sex_metrics_3, "sex")

    # Run with custom y_lim and adjusted thresholds
    eqb.eq_disparity_metrics_point_plot(
        dispa=[race_metrics_3, sex_metrics_3],
        metric_cols=["Accuracy", "Precision", "Recall"],
        category_names=["race", "sex"],
        figsize=(6, 8),
        include_legend=True,
        disparity_thresholds=(0.9, 1.1),
        show_pass_fail=True,
        show_grid=True,
        y_lim=(0.7, 1.3),
    )
