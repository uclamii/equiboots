import numpy as np
import pandas as pd
from equiboots import EquiBoots
from sklearn.preprocessing import MultiLabelBinarizer
import equiboots as eqb


def eq_general_test(task):
    if task == "binary_classification":
        n_classes = 2
        n_samples = 1000
        y_prob = np.random.RandomState(3).rand(n_samples)
        y_pred = (y_prob > 0.5) * 1
        y_true = np.random.RandomState(30).randint(0, n_classes, n_samples)
    elif task == "multi_class_classification":
        n_classes = 3
        n_samples = 1000
        y_prob = np.random.RandomState(3).rand(n_samples, n_classes)
        y_prob /= y_prob.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.random.RandomState(30).randint(0, n_classes, n_samples)
    elif task == "regression":
        n_classes = 3
        n_samples = 1000
        y_true = np.random.RandomState(3).rand(n_samples)
        y_pred = np.random.RandomState(30).rand(n_samples)
        y_prob = None
    elif task == "multi_label_classification":
        n_classes = 3
        n_samples = 7000
        # need to specify seeds for reproducibility
        y_true = [
            np.random.RandomState(seed + 1).choice(
                range(n_classes),
                size=np.random.RandomState(seed).randint(1, n_classes + 1),
                replace=False,
            )
            for seed, _ in enumerate(range(n_samples))
        ]
        # one-hot encode sequences
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(y_true)
        y_prob = np.random.RandomState(3).rand(n_samples, n_classes)  # 3 classes
        y_prob /= y_prob.sum(axis=1, keepdims=True)
        y_pred = (y_prob > 0.5) * 1
    else:
        raise ValueError("Invalid task")

    # fix seed for reproducibility
    race = (
        np.random.RandomState(3)
        .choice(["white", "black", "asian", "hispanic"], n_samples)
        .reshape(-1, 1)
    )
    sex = np.random.RandomState(31).choice(["M", "F"], n_samples).reshape(-1, 1)
    fairness_df = pd.DataFrame(
        data=np.concatenate((race, sex), axis=1), columns=["race", "sex"]
    )

    eq = EquiBoots(
        y_true,
        y_prob,
        y_pred,
        fairness_df,
        fairness_vars=["race", "sex"],
        reference_groups=["white", "M"],
        task=task,
        bootstrap_flag=True,
        num_bootstraps=10,
        boot_sample_size=100,
        balanced=True,  # False is stratified, True is balanced
        stratify_by_outcome=False,
        group_min_size=50,
    )

    # Set group_min_size based on task for debugging purposes
    # regression: group_min_size = 240
    # binary_classification: group_min_size = 120
    # multi_class_classification: group_min_size = 70
    # multi_label_classification: group_min_size = 1750

    # Set seeds
    eq.set_fix_seeds([42, 123, 222, 999])

    print("seeds", eq.seeds)

    eq.grouper(groupings_vars=["race", "sex"])

    print("groups", eq.groups)

    data = eq.slicer("race")

    for key in data[0].keys():
        print("key", key)
        print(data[0][key]["y_true"].shape)
        print(np.unique(data[0][key]["y_true"], axis=0, return_counts=True))

    print("Categories below minimum size", eq.groups_below_min_size)

    # The metrics are calculated for each group in the groupings_vars
    race_metrics = eq.get_metrics(data)

    print("race_metrics", race_metrics)
    print("len(race_metrics)", len(race_metrics))

    dispa = eq.calculate_disparities(race_metrics, "race")

    # Create DataFrame from disparities
    disa_metrics_df = eqb.metrics_dataframe(metrics_data=dispa)
    print(f"Disparity Metrics DataFrame\n{disa_metrics_df}\n")

    print("dispa", dispa)
    print("len(dispa)", len(dispa))

    # Calculate differences

    diffs = eq.calculate_differences(race_metrics, "race")
    
    # Create DataFrame from differences
    disa_diffs_df = eqb.metrics_dataframe(metrics_data=diffs)
    print(f"Disparity Metrics DataFrame\n{disa_diffs_df}\n")

    print("diffs", diffs)
    print("len(diffs)", len(diffs))


if __name__ == "__main__":
    task = "binary_classification"
    eq_general_test(task)
