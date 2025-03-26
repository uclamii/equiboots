import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from .metrics import (
    multi_class_classification_metrics,
    binary_classification_metrics,
    multi_label_classification_metrics,
    regression_metrics,
)


class EquiBoots:

    def __init__(
        self,
        y_true: np.array,
        y_prob: np.array,
        y_pred: np.array,
        fairness_df: pd.DataFrame,
        fairness_vars: list,
        seeds: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        reference_groups: list = None,
        task: str = "binary_classification",
        bootstrap_flag: bool = False,
        num_bootstraps: int = 10,
        boot_sample_size: int = 100,
        balanced: bool = True,  # sample balanced or stratified
    ) -> None:

        self.fairness_vars = fairness_vars
        self.task = task
        self.y_true = y_true
        self.y_prob = y_prob
        self.y_pred = y_pred
        self.fairness_df = fairness_df
        self.groups = {}
        self.seeds = seeds
        self.check_task(task)
        self.check_fairness_vars(fairness_vars)
        self.set_reference_groups(reference_groups)
        self.bootstrap_flag = bootstrap_flag
        self.num_bootstraps = num_bootstraps
        self.boot_sample_size = boot_sample_size
        self.balanced = balanced

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
        """
        Groups data by categorical variables and stores indices for each category.

        Parameters:
        groupings_vars : list
            Categorical variables to group by.

        Returns:
        None
        """
        if self.bootstrap_flag:
            self.groups = self.bootstrap(
                groupings_vars=groupings_vars,
                seeds=self.seeds,
                n_iterations=self.num_bootstraps,
                sample_size=self.boot_sample_size,
                balanced=self.balanced,
            )
            print("Groups created")
            return
        else:
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

    def bootstrap(
        self,
        seeds: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_iterations: int = 2,
        sample_size: int = 10,
        groupings_vars: list = None,
        balanced: bool = True,
    ):
        """
        Perform balanced bootstrap sampling on the dataset.

        Parameters:
        seeds : list
            List of random seeds for reproducibility.
        n_iterations : int
            Number of bootstrap iterations.
        sample_size : int
            Size of each bootstrap sample.
        groupings_vars : list
            Variables to group by during sampling.
        balanced : bool
            Whether to balance samples across groups.

        Returns:
        list
            List of bootstrapped samples with group indices.
        """

        bootstrapped_samples = []
        for indx in tqdm(range(n_iterations), desc="Bootstrapping iterations"):
            groups = {}

            for var in groupings_vars:
                categories = self.fairness_df[var].unique()
                n_categories = len(categories)
                groups[var] = {}
                groups[var]["categories"] = categories
                groups[var]["indices"] = {}

                for cat in categories:

                    group = self.fairness_df[self.fairness_df[var] == cat].index

                    if balanced:
                        n_samples = max(1, int(sample_size / n_categories))
                    else:
                        n_samples = max(
                            1, int(len(group) * sample_size / len(self.fairness_df))
                        )

                    sampled_group = resample(
                        group,
                        replace=True,
                        n_samples=n_samples,
                        random_state=seeds[indx % len(seeds)],
                    )

                    groups[var]["indices"][cat] = sampled_group
            bootstrapped_samples.append(groups)

        return bootstrapped_samples

    def slicer(self, slicing_var: str) -> pd.DataFrame:
        """
        Slices y_true, y_prob, and y_pred by a categorical variable, with or
        without bootstrapping.

        Parameters:
        slicing_var : str
            The categorical variable to slice by.

        Returns:
        list of dictionaries or dictionary
            Sliced data grouped by the variable's categories.
        """

        if self.bootstrap_flag:
            return [self.groups_slicer(groups, slicing_var) for groups in self.groups]
        else:
            return self.groups_slicer(self.groups, slicing_var)

    def groups_slicer(self, groups, slicing_var: str) -> pd.DataFrame:
        """
        Slices y_true, y_prob, and y_pred into categories of a given variable.

        Parameters:
        groups : dict
            Group indices for slicing.
        slicing_var : str
            The categorical variable to slice by.

        Returns:
        dictionary
            Sliced data grouped by categories.
        """

        data = {}
        categories = groups[slicing_var]["categories"]
        for cat in categories:
            if self.task in [
                "binary_classification",
                "multi_label_classification",
                "multi_class_classification",
            ]:
                y_true = self.y_true[groups[slicing_var]["indices"][cat]]
                y_prob = self.y_prob[groups[slicing_var]["indices"][cat]]
                y_pred = self.y_pred[groups[slicing_var]["indices"][cat]]
                data[cat] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}
            elif self.task == "regression":
                y_true = self.y_true[groups[slicing_var]["indices"][cat]]
                y_pred = self.y_pred[groups[slicing_var]["indices"][cat]]
                data[cat] = {"y_true": y_true, "y_pred": y_pred}

        return data

    def get_metrics(self, sliced_dict) -> dict:
        """Calculate metrics for each group based on the task type."""
        if self.bootstrap_flag:
            return [self.get_groups_metrics(sliced) for sliced in sliced_dict]
        else:
            return self.get_groups_metrics(sliced_dict)

    def get_groups_metrics(self, sliced_dict: dict) -> dict:
        """Calculate metrics for each group based on the task type."""
        sliced_dict_metrics = {}

        for group, data in sliced_dict.items():

            if self.task == "binary_classification":
                y_true = data["y_true"]
                y_prob = data["y_prob"]
                y_pred = data["y_pred"]
                metrics = binary_classification_metrics(
                    y_true,
                    y_pred,
                    y_prob,
                )
            elif self.task == "multi_class_classification":
                y_true = data["y_true"]
                y_prob = data["y_prob"]
                y_pred = data["y_pred"]
                n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
                metrics = multi_class_classification_metrics(
                    y_true, y_pred, y_prob, n_classes
                )
            elif self.task == "multi_label_classification":
                y_true = data["y_true"]
                y_prob = data["y_prob"]
                y_pred = data["y_pred"]
                metrics = multi_label_classification_metrics(
                    y_true,
                    y_pred,
                    y_prob,
                )
            elif self.task == "regression":
                y_true = data["y_true"]
                y_pred = data["y_pred"]
                metrics = regression_metrics(y_true, y_pred)

            sliced_dict_metrics[group] = metrics

        return sliced_dict_metrics

    def calculate_disparities(self, metric_dict, var_name: str) -> dict:
        """Calculate metrics for each group based on the task type."""
        if self.bootstrap_flag:
            return [
                self.calculate_groups_disparities(metrics, var_name=var_name)
                for metrics in metric_dict
            ]
        else:
            return self.calculate_groups_disparities(
                metric_dict,
                var_name=var_name,
            )

    def calculate_groups_disparities(self, metric_dict: dict, var_name: str) -> dict:
        """
        Calculate disparities between each group and the reference group.
        """
        # metric_dict = self.get_metrics(sliced_dict)
        ref_cat = self.reference_groups[var_name]
        disparities = {}

        # Get reference group metrics
        ref_metrics = metric_dict[ref_cat]

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
