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

    eq2 = eqb.EquiBoots(
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
    int_list = np.linspace(0, 100, num=10, dtype=int).tolist()

    eq2.set_fix_seeds(int_list)

    print("seeds", eq2.seeds)

    eq2.grouper(groupings_vars=["race", "sex"])

    data = eq2.slicer("race")
    race_metrics = eq2.get_metrics(data)

    dispa = eq2.calculate_disparities(race_metrics, "race")

    melted = pd.DataFrame(dispa).melt()
    df = melted["value"].apply(pd.Series).assign(attribute_value=melted["variable"])

    eqb.eq_disparity_metrics_plot(
        dispa,
        metric_cols=[
            "Accuracy_ratio",
            "Precision_ratio",
            "Predicted Prevalence_ratio",
            "FP Rate_ratio",
            "TN Rate_ratio",
            "Recall_ratio",
        ],
        name="race",
        categories="all",
        figsize=(24, 4),
        plot_kind="violinplot",
        color_by_group=True,
        show_grid=False,
        strict_layout=True,
        save_path="./images",
        show_pass_fail=True,
        # y_lim=(-2, 4),
        # plot_thresholds=[0.9, 1.2],
    )

    eqb.eq_plot_bootstrapped_group_curves(
        boot_sliced_data=data,
        curve_type="roc",
        title="Bootstrapped ROC Curve by Race",
        filename="boot_roc_race",
        # bar_every=100,
        dpi=100,
        n_bins=10,
        figsize=(6, 6),
        color_by_group=True,
    )
