import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy.stats.contingency import association


@dataclass
class StatTestResult:
    """Stores statistical test results including test statistic, p-value, and significance."""

    statistic: float
    p_value: float
    is_significant: bool
    test_name: str
    critical_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        # Ensure is_significant is a Python bool
        self.is_significant = bool(self.is_significant)


class StatisticalTester:
    """Performs statistical significance testing on metrics with support for various tests and data types."""

    AVAILABLE_TESTS = {
        "chi_square": "Chi-square test",
        "bootstrap_test": "Bootstrap test",
    }

    ADJUSTMENT_METHODS = {
        "bonferroni": "Bonferroni correction",
        "fdr_bh": "Benjamini-Hochberg FDR",
        "holm": "Holm-Bonferroni",
        "none": "No correction",
    }

    def __init__(self):
        """Initializes StatisticalTester with default test implementations."""
        self._test_implementations = {
            "chi_square": self._chi_square_test,
            "bootstrap_test": self._bootstrap_test,
        }

    def _bootstrap_test(self, data: List[float]) -> List[float]:

        # 3. 95% C.I. for a sample
        mu = np.mean(data)
        sigma = np.std(data)
        se = sigma / np.sqrt(len(data))

        ## Lower and Upper Bounds of C.I.
        ci_lower, ci_upper = stats.norm.interval(0.95, loc=mu, scale=se)

        # 4. Check if C.I. overlaps 0, if yes non statistically significant

        # Does CI cross zero?
        if ci_lower <= 0 <= ci_upper:
            is_significant = False
        else:
            is_significant = True

        # 5. Calculate p_value, and asjust_p, effect size using bootstrap (optional)
        # two‐sided p‐value under H0: mean=0
        z = mu / se if se else np.nan
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # 6. Return StatisticalTestResult object
        return StatTestResult(
            statistic=mu,
            p_value=p_value,
            is_significant=is_significant,
            test_name="bootstrap_mean",
            confidence_interval=(ci_lower, ci_upper),
            # effect_size=effect_size,
        )

    def _chi_square_test(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
    ) -> StatTestResult:
        """Performs Chi-square test for categorical data.

        Args:
            metrics: Metrics of CM in a dictionary
            config: Configuration dictionary containing test parameters

        Returns:
            StatTestResult object containing test results
        """
        # Convert to numpy arrays
        data = pd.DataFrame(metrics)
        # Create contingency table
        contingency_table = data.T

        # Use scipy's implementation
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

        return StatTestResult(
            statistic=chi2,
            p_value=p_value,
            is_significant=p_value < config.get("alpha", 0.05),
            test_name="Chi-Square Test",
        )

    def _calculate_effect_size(self, metrics: Dict) -> float:
        """Calculates the Cramer's V effect size using scipy.

        Returns:
            float: Cramer's V effect size
        """
        coningency_table = pd.DataFrame(metrics).T
        effect_size = association(coningency_table, method="cramer")
        return effect_size

    def _adjust_p_values(
        self, results: Dict[str, Dict[str, StatTestResult]], method: str, alpha: float
    ) -> Dict[str, Dict[str, StatTestResult]]:
        """Adjusts p-values for multiple comparisons using specified method."""

        p_values = []
        for group_results in results.values():
            p_values.append(group_results.p_value)

        if method == "bonferroni":
            adjusted_p_values = multipletests(
                p_values, alpha=alpha, method="bonferroni"
            )[1]
        elif method == "fdr_bh":
            adjusted_p_values = multipletests(p_values, alpha=alpha, method="fdr_bh")[1]
        elif method == "holm":
            adjusted_p_values = multipletests(p_values, alpha=alpha, method="holm")[1]
        else:
            return results

        for idx, group in enumerate(results.keys()):
            results[group].p_value = adjusted_p_values[idx]
            results[group].is_significant = adjusted_p_values[idx] < alpha

        return results

    def analyze_metrics(
        self,
        metrics_data: Union[Dict, List[Dict]],
        reference_group: str,
        test_config: Dict[str, Any],
        task: Optional[str] = None,
        differences: Optional[dict] = None,
    ) -> Dict[str, Dict[str, StatTestResult]]:
        """Analyzes metrics for statistical significance against a reference group."""

        config = {**test_config}
        self._validate_config(config)

        if isinstance(metrics_data, list):
            results = self._analyze_bootstrapped_metrics(
                differences, reference_group, config
            )
        else:
            if task == "binary_classification":
                results = self._analyze_single_metrics(
                    metrics_data, reference_group, config
                )
                ## TODO make adjust to method
                # adjust p_values
                if config["adjust_method"] != "none":
                    # Avoid running this command if results have a len of 1; then
                    # we do not need to adj. p-value
                    if len(results) > 1:
                        # Adjust p-values for multiple comparisons
                        adjusted_results = self._adjust_p_values(
                            results, config["adjust_method"], config["alpha"]
                        )
                        results = adjusted_results
                ##
            else:
                raise ValueError(
                    "Task not supported for non-bootstrapped metrics. "
                    "Use bootstrapped metrics."
                )

        return results

    def _validate_config(self, config: Dict[str, Any]):
        """Validates the configuration dictionary for required keys and values."""
        required_keys = ["test_type", "alpha"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        if config["test_type"] not in self.AVAILABLE_TESTS:
            raise ValueError(
                f"Invalid test type: {config['test_type']}. Available tests: {self.AVAILABLE_TESTS.keys()}"
            )

        if config["adjust_method"] not in self.ADJUSTMENT_METHODS:
            raise ValueError(
                f"Invalid adjustment method: {config['adjust_method']}. Available methods: {self.ADJUSTMENT_METHODS.keys()}"
            )

    def _analyze_single_metrics(
        self, metrics: Dict, reference_group: str, config: Dict[str, Any]
    ) -> Dict[str, Dict[str, StatTestResult]]:
        """Analyzes non-bootstrapped metrics against a reference group."""

        results = {}

        test_func = self._test_implementations[config["test_type"]]

        metrics_CM = ["TP", "FP", "TN", "FN"]
        # Get the keys of the metrics dictionary

        metrics = {
            key: {k: v for k, v in metrics[key].items() if k in metrics_CM}
            for key in metrics.keys()
        }

        ref_metrics = {k: v for k, v in metrics.items() if k in [reference_group]}

        # omnibus test
        results["omnibus"] = test_func(metrics, config)

        if results["omnibus"].is_significant:
            effect_size = self._calculate_effect_size(metrics)
            results["omnibus"].effect_size = effect_size
            for group, _ in metrics.items():
                if group == reference_group:
                    continue

                comp_metrics = {k: v for k, v in metrics.items() if k in [group]}

                ref_comp_metrics = {**ref_metrics, **comp_metrics}

                results[group] = test_func(ref_comp_metrics, config)
                if results[group].is_significant:
                    effect_size = self._calculate_effect_size(ref_comp_metrics)
                    results[group].effect_size = effect_size
                else:
                    results[group].effect_size = None
                    results[group].confidence_interval = None

            return results

        else:  # no need to calculate effect size
            results["omnibus"].effect_size = None
            results["omnibus"].confidence_interval = None
            # no need for pairwise test
            return results

    def _analyze_bootstrapped_metrics(
        self, metrics_diff: list[Dict], reference_group: str, config: Dict[str, Any]
    ) -> Dict[str, Dict[str, StatTestResult]]:
        """Analyzes bootstrapped metrics differences against a reference group."""

        results = {}

        test_func = self._test_implementations[config["test_type"]]

        metrics_boot = ["Accuracy_diff", "Precision_diff"]

        aggregated_metric_dict = {}
        for metric_dict in metrics_diff:
            ## getting rid of reference group
            metric_dict.pop(reference_group, None)

            for group_key, group_metrics in metric_dict.items():
                ### create new key in dictionary for each group e.g. "asian" and set to an empty list
                if group_key not in aggregated_metric_dict:
                    aggregated_metric_dict[group_key] = {
                        metric: [] for metric in metrics_boot
                    }

                ## populate list with the values from each bootstrap for each group
                for metric in metrics_boot:
                    aggregated_metric_dict[group_key][metric].append(
                        group_metrics[metric]
                    )

        # calls test for each group e.g. hispanic etc. and then calls the
        # bootstrap test func for each metric. e.g. Precision_diff
        for group_key, group_metrics in aggregated_metric_dict.items():
            results[group_key] = {}
            for metric in metrics_boot:
                test_result = test_func(group_metrics[metric])
                results[group_key][metric] = test_result

        # 7. return for List[Dict[str=group,Dict[str=metric,StatisticalTestResult]]]
        return results
