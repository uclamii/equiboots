import pandas as pd
import numpy as np
from metrics import (
    binary_classification_metrics,
    multi_class_prevalence,
    multi_label_classification_metrics,
    regression_metrics,
)
from scipy import stats
import itertools


class EquiBoots:

    def __init__(
        self,
        y_true: np.array,
        y_prob: np.array,
        y_pred: np.array,
        fairness_df: pd.DataFrame,
        fairness_vars: list,
        reference_groups: list = None,
        task: str = "binary_classification",
    ) -> None:

        self.fairness_vars = fairness_vars
        self.task = task
        self.y_true = y_true
        self.y_prob = y_prob
        self.y_pred = y_pred
        self.fairness_df = fairness_df
        self.groups = {}
        self.seeds = []
        self.check_task(task)
        self.check_fairness_vars(fairness_vars)
        self.set_reference_groups(reference_groups)
        pass

    def set_reference_groups(self, reference_groups):
        ### zip the reference groups
        if reference_groups:
            self.reference_groups = dict(zip(self.fairness_vars, reference_groups))
        else:
            ### use the most populous group as the reference group if needed
            self.reference_groups = {}
            for var in self.fairness_vars:
                value_counts = self.fairness_df[var].value_counts()
                self.reference_groups[var] = value_counts.index[0]

    def check_task(self, task):
        if task not in [
            "binary_classification",
            "multi_class_classification",
            "regression",
            "multi_label_classification",
        ]:
            raise ValueError(
                f"Invalid task, please supply one of 'binary_classification', "
                f"'multi_class_classification', 'regression' or 'multi_label_classification'"
            )

    def check_fairness_vars(self, fairness_vars):
        if fairness_vars is None:
            raise ValueError("fairness_vars cannot be None, please provide a list")
        if not isinstance(fairness_vars, list):
            raise ValueError("fairness_vars must be a list")

    def grouper(self, groupings_vars: list) -> pd.DataFrame:
        """Method that given a list of categorical variables, returns indices of each category."""
        for var in groupings_vars:
            self.groups[var] = {}
            # Replace NaN with 'missing' to treat missing values as a category
            self.fairness_df[var] = self.fairness_df[var].fillna("missing")
            self.groups[var]["categories"] = self.fairness_df[var].unique()
            self.groups[var]["indices"] = {}
            for cat in self.groups[var]["categories"]:
                self.groups[var]["indices"][cat] = self.fairness_df[
                    self.fairness_df[var] == cat
                ].index
        print("Groups created")
        return

    def slicer(self, slicing_var: str) -> pd.DataFrame:
        """Method that given a categorical variable,
        slices the y_true and y_prob into the different categories of the variable"""
        data = {}
        categories = self.groups[slicing_var]["categories"]
        for cat in categories:
            y_true = self.y_true[self.groups[slicing_var]["indices"][cat]]
            y_prob = self.y_prob[self.groups[slicing_var]["indices"][cat]]
            y_pred = self.y_pred[self.groups[slicing_var]["indices"][cat]]
            data[cat] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}
        return data

    def get_metrics(self, sliced_dict: dict) -> dict:
        """Calculate metrics for each group based on the task type."""
        metric_sliced_dict = {}

        for group, data in sliced_dict.items():
            y_true = data["y_true"]
            y_prob = data["y_prob"]
            y_pred = data["y_pred"]

            if self.task == "binary_classification":
                metrics = binary_classification_metrics(y_true, y_pred, y_prob)
            elif self.task == "multi_class_classification":
                n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
                metrics = multi_class_prevalence(y_true, y_pred, n_classes)
            elif self.task == "multi_label_classification":
                metrics = multi_label_classification_metrics(y_true, y_pred, y_prob)
            elif self.task == "regression":
                metrics = regression_metrics(y_true, y_pred)

            metric_sliced_dict[group] = metrics

        return metric_sliced_dict

    def calculate_disparities(self, sliced_dict: dict, group: str) -> dict:
        """
        Calculate disparities between each group and the reference group.
        """
        metric_dict = self.get_metrics(sliced_dict)
        ref_group = self.reference_groups[group]
        disparities = {}

        # Get reference group metrics
        ref_metrics = metric_dict[ref_group]

        # Calculate disparities for each group
        for category, metrics in metric_dict.items():

            disparities[category] = {}
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)) or not isinstance(
                    ref_metrics.get(metric_name), (int, float)
                ):
                    continue

                ref_value = ref_metrics[metric_name]
                if ref_value != 0:
                    ratio = value / ref_value
                    disparities[category][f"{metric_name}_ratio"] = ratio
                else:
                    disparities[category][f"{metric_name}_ratio"] = -1
                    raise Warning(
                        "Reference metric value is zero returning negative value"
                    )

        return disparities

    def set_fix_seeds(self, seeds: list) -> None:
        """
        Set fixed random seeds for bootstrapping or reproducibility.
        """
        if not all(isinstance(seed, int) for seed in seeds):
            raise ValueError("All seeds must be integers.")
        self.seeds = seeds


if __name__ == "__main__":
    # Test the class
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

    eq = EquiBoots(
        y_true,
        y_prob,
        y_pred,
        fairness_df,
        fairness_vars=["race", "sex"],
        # reference_groups=["white", "M"],
    )

    eq.grouper(groupings_vars=["race", "sex"])

    data = eq.slicer("race")
    race_metrics = eq.get_metrics(data)

    dispa = eq.calculate_disparities(data, "race")

    print(dispa)

    # Set seeds
    eq.set_fix_seeds([42, 123, 999])

    print(eq.seeds)
