import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.utils import resample
from .metrics import (
    multi_class_classification_metrics,
    binary_classification_metrics,
    multi_label_classification_metrics,
    regression_metrics,
)
from .StatisticalTester import StatisticalTester, StatTestResult
from typing import Optional, Dict, Any, List


class EquiBoots:

    def __init__(
        self,
        y_true: np.array,
        y_pred: np.array,
        fairness_df: pd.DataFrame,
        fairness_vars: list,
        y_prob: np.array = None,
        seeds: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        reference_groups: list = None,
        task: str = "binary_classification",
        bootstrap_flag: bool = False,
        num_bootstraps: int = 10,
        boot_sample_size: int = 100,
        balanced: bool = True,  # sample balanced or stratified
        stratify_by_outcome: bool = False,  # sample stratified by outcome
        group_min_size: int = 10,
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
        self.stratify_by_outcome = stratify_by_outcome
        self.group_min_size = group_min_size
        self.groups_below_min_size = {var: set() for var in fairness_vars}

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
                f"'multi_class_classification', "
                f"'regression' or 'multi_label_classification'"
            )

    def check_classification_task(self, task):
        if task == "regression":
            raise ValueError(
                f"Invalid task, please supply one of 'binary_classification', "
                f"'multi_class_classification' or 'multi_label_classification'"
                f" to stratify by outcome."
            )

    def check_fairness_vars(self, fairness_vars):
        if fairness_vars is None:
            raise ValueError("fairness_vars cannot be None, please provide a list")
        if not isinstance(fairness_vars, list):
            raise ValueError("fairness_vars must be a list")

    def check_group_size(self, group: pd.Index, cat: str, var: str) -> bool:
        """
        Check if a group meets the minimum size requirement.
        """
        if len(group) < self.group_min_size:
            warnings.warn(
                f"Group '{cat}' for variable '{var}' has less than "
                f"{self.group_min_size} samples. Skipping category of this group."
            )
            self.groups_below_min_size[var].add(cat)
            return False
        return True

    def check_group_empty(self, sampled_group: np.array, cat: str, var: str) -> bool:
        """
        Check if sampled group is empty.
        """
        if sampled_group.size == 0:
            warnings.warn(
                f"Sampled Group '{cat}' for variable '{var}' has no samples. "
                f"Skipping category of this group."
            )
            return False
        return True

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
                    group = self.fairness_df[self.fairness_df[var] == cat].index
                    # Check if the group size is less than the minimum size
                    if not self.check_group_size(group, cat, var):
                        continue
                    # Store the indices of the group
                    self.groups[var]["indices"][cat] = group
            print("Groups created")
            return

    def bootstrap(
        self,
        seeds: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_iterations: int = 2,
        sample_size: int = 10,
        groupings_vars: list = None,
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

                    if self.stratify_by_outcome:

                        self.check_classification_task(self.task)

                        sampled_group = np.array([])
                        # Stratified sampling by outcome
                        for outcome in np.unique(self.y_true, axis=0):

                            # Create a mask for the current category and outcome
                            if self.y_true.ndim == 1:
                                # 1D case: Direct comparison
                                slicing_condition = (self.fairness_df[var] == cat) & (
                                    self.y_true == outcome
                                )
                            else:
                                # Multi-dimensional case: Row-wise comparison
                                slicing_condition = (self.fairness_df[var] == cat) & (
                                    np.all(self.y_true == outcome, axis=1)
                                )

                            group = self.fairness_df[slicing_condition].index

                            # Check if the group size is less than the minimum size
                            if not self.check_group_size(group, cat, var):
                                continue

                            # Sample from the group and concatenate
                            sampled_group = np.concatenate(
                                (
                                    sampled_group,
                                    self.sample_group(
                                        group,
                                        n_categories,
                                        indx,
                                        sample_size,
                                        seeds,
                                        self.balanced,
                                    ),
                                )
                            ).astype(int)
                    else:
                        # Regular sampling
                        group = self.fairness_df[self.fairness_df[var] == cat].index
                        # Check if the group size is less than the minimum size
                        if not self.check_group_size(group, cat, var):
                            continue
                        sampled_group = self.sample_group(
                            group,
                            n_categories,
                            indx,
                            sample_size,
                            seeds,
                            self.balanced,
                        )

                    # Check if the sampled group is empty
                    if not self.check_group_empty(sampled_group, cat, var):
                        continue
                    # Store the sampled group indices
                    groups[var]["indices"][cat] = sampled_group
            bootstrapped_samples.append(groups)

        return bootstrapped_samples

    def sample_group(self, group, n_categories, indx, sample_size, seeds, balanced):
        """
        Samples a group with or without balancing.

        Parameters:
        group : array-like
            Indices of the group to sample from.
        n_categories : int
            Number of categories in the grouping variable.
        indx : int
            Current iteration index for seed selection.
        sample_size : int
            Total sample size for bootstrapping.
        seeds : list
            List of random seeds for reproducibility.
        balanced : bool
            Whether to balance samples across categories.

        Returns:
        array
            Sampled indices from the group.
        """

        if balanced:
            n_samples = max(1, int(sample_size / n_categories))
        else:
            n_samples = max(1, int(len(group) * sample_size / len(self.fairness_df)))

        sampled_group = resample(
            group,
            replace=True,
            n_samples=n_samples,
            random_state=seeds[indx % len(seeds)],
        )
        return sampled_group

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
            if cat in self.groups_below_min_size[slicing_var]:
                warnings.warn(
                    f"Group '{cat}' for variable '{slicing_var}' has less than "
                    f"{self.group_min_size} samples. "
                    f"Skipping catategory of this group."
                )
                continue

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
            print("Calculating metrics for each bootstrap:")
            return [self.get_groups_metrics(sliced) for sliced in tqdm(sliced_dict)]
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
        """Calculate disparities metrics for each group based on the task type."""
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
                sanitized_name = metric_name.replace(" ", "_")
                if ref_value != 0:
                    ratio = value / ref_value
                    disparities[category][f"{sanitized_name}_Ratio"] = ratio
                else:
                    disparities[category][f"{sanitized_name}_Ratio"] = -1
                    warnings.warn(
                        "Reference metric value is zero returning negative value"
                    )

        return disparities

    def calculate_differences(self, metric_dict, ref_var_name: str) -> dict:
        """Calculate difference metrics for each group based on the task type."""
        if self.bootstrap_flag:
            return [
                self.calculate_groups_differences(metrics, ref_var_name=ref_var_name)
                for metrics in metric_dict
            ]
        else:
            return self.calculate_groups_differences(
                metric_dict,
                ref_var_name=ref_var_name,
            )

    def calculate_groups_differences(
        self, metric_dict: dict, ref_var_name: str
    ) -> dict:
        """
        Calculate differences between each group and the reference group.
        """
        ref_cat = self.reference_groups[ref_var_name]
        differences = {}

        # Get reference group metrics
        ref_metrics = metric_dict[ref_cat]

        # Calculate disparities for each group
        for category, metrics in metric_dict.items():

            differences[category] = {}
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)) or not isinstance(
                    ref_metrics.get(metric_name), (int, float)
                ):
                    continue

                ref_value = ref_metrics[metric_name]
                sanitized_name = metric_name.replace(" ", "_")
                difference = value - ref_value
                differences[category][f"{sanitized_name}_diff"] = difference

        return differences

    def set_fix_seeds(self, seeds: list) -> None:
        """
        Set fixed random seeds for bootstrapping or reproducibility.
        """
        if not all(isinstance(seed, int) for seed in seeds):
            raise ValueError("All seeds must be integers.")
        self.seeds = seeds

    def analyze_statistical_significance(
        self,
        metric_dict: dict,
        var_name: str,
        test_config: Dict[str, Any],
        differences: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict[str, StatTestResult]]:
        """Analyzes statistical significance of metric differences between groups.

        Args:
            metric_dict: Dictionary of metrics from get_metrics()
            var_name: Name of the demographic variable being analyzed
            test_config: Optional configuration for statistical testing:
                - test_type: Type of test (chi-squared, bootstrap test)
                - alpha: Significance level (default: 0.05)
                - adjust_method: Multiple comparison adjustment (bonferroni, fdr_bh, holm, none)
                - confidence_level: Confidence level for intervals (default: 0.95)
                - classification_task: Whether the task is classification (default: True)


        Returns:
            Dictionary containing test results for each group and metric, with StatTestResult objects
            containing:
            - test statistics
            - p-values (adjusted if specified)
            - significance flags
            - effect sizes (Cohen's d for t-test, rank-biserial correlation for non-parametric tests)
            - confidence intervals (where applicable)
        """
        tester = StatisticalTester()
        reference_group = self.reference_groups[var_name]

        if test_config is None:
            raise ValueError("test_config cannot be None, please provide a dictionary")

        test_results = tester.analyze_metrics(
            metrics_data=metric_dict,
            reference_group=reference_group,
            test_config=test_config,
            task=self.task,
            differences=differences,
        )

        return test_results

    @staticmethod
    def list_available_tests() -> Dict[str, str]:
        """List available statistical tests and their descriptions."""
        return StatisticalTester.AVAILABLE_TESTS

    @staticmethod
    def list_adjustment_methods() -> Dict[str, str]:
        """List available adjustment methods and their descriptions."""
        return StatisticalTester.ADJUSTMENT_METHODS
