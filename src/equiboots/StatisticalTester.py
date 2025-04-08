import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class StatTestResult:
    """Stores statistical test results including test statistic, p-value, and significance."""
    statistic: float
    p_value: float
    is_significant: bool
    test_name: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

class StatisticalTester:
    """Performs statistical significance testing on metrics with support for various tests and data types."""
    
    AVAILABLE_TESTS = {
        "mann_whitney": "Mann-Whitney U test (non-parametric)",
        "t_test": "Welch's t-test (parametric)",
        "ks_test": "Kolmogorov-Smirnov test",
        "bootstrap_test": "Bootstrap-based permutation test",
        "wilcoxon": "Wilcoxon signed-rank test",
        "permutation": "Custom permutation test"
    }
    
    ADJUSTMENT_METHODS = {
        "bonferroni": "Bonferroni correction",
        "fdr_bh": "Benjamini-Hochberg FDR",
        "holm": "Holm-Bonferroni",
        "none": "No correction"
    }

    def __init__(self):
        """Initializes StatisticalTester with default test implementations."""
        self._test_implementations = {
            "mann_whitney": self._mann_whitney_test,
            "t_test": self._t_test,
            "ks_test": self._ks_test,
            "bootstrap_test": self._bootstrap_test,
            "wilcoxon": self._wilcoxon_test,
            "permutation": self._permutation_test
        }

    def _mann_whitney_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any]) -> StatTestResult:
        """Performs Mann-Whitney U test for non-parametric comparison."""
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
            
        statistic, p_value = stats.mannwhitneyu(
            ref_data,
            comp_data,
            alternative=config.get("alternative", "two-sided")
        )
        
        effect_size = self._calculate_effect_size(ref_data, comp_data)
        
        return StatTestResult(
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < config.get("alpha", 0.05),
            test_name="Mann-Whitney U",
            effect_size=effect_size
        )

    def _t_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any]) -> StatTestResult:
        """Performs Welch's t-test for parametric comparison."""
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
            
        statistic, p_value = stats.ttest_ind(
            ref_data,
            comp_data,
            equal_var=False,
            alternative=config.get("alternative", "two-sided")
        )
        
        effect_size = self._calculate_effect_size(ref_data, comp_data)
        
        return StatTestResult(
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < config.get("alpha", 0.05),
            test_name="Welch's t-test",
            effect_size=effect_size
        )

    def _ks_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any]) -> StatTestResult:
        """Performs Kolmogorov-Smirnov test for distribution comparison."""
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
            
        statistic, p_value = stats.ks_2samp(ref_data, comp_data)
        
        effect_size = self._calculate_effect_size(ref_data, comp_data)
        
        return StatTestResult(
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < config.get("alpha", 0.05),
            test_name="Kolmogorov-Smirnov",
            effect_size=effect_size
        )

    def _calculate_effect_size(self, ref_data: List[float], comp_data: List[float]) -> float:
        """Calculates Cohen's d effect size using scipy.stats."""
        from scipy.stats import cohen_d
        return cohen_d(comp_data, ref_data)

    def _adjust_p_values(self, results: Dict[str, Dict[str, StatTestResult]], method: str, alpha: float) -> Dict[str, Dict[str, StatTestResult]]:
        """Adjusts p-values for multiple comparisons using specified method."""
        p_values = []
        for group_results in results.values():
            for test_result in group_results.values():
                p_values.append(test_result.p_value)
        
        if method == "bonferroni":
            adjusted_p_values = multipletests(p_values, alpha=alpha, method='bonferroni')[1]
        elif method == "fdr_bh":
            adjusted_p_values = multipletests(p_values, alpha=alpha, method='fdr_bh')[1]
        elif method == "holm":
            adjusted_p_values = multipletests(p_values, alpha=alpha, method='holm')[1]
        else:
            return results
        
        p_value_idx = 0
        adjusted_results = {}
        for group, group_results in results.items():
            adjusted_results[group] = {}
            for metric, test_result in group_results.items():
                adjusted_result = StatTestResult(
                    statistic=test_result.statistic,
                    p_value=adjusted_p_values[p_value_idx],
                    is_significant=adjusted_p_values[p_value_idx] < alpha,
                    test_name=test_result.test_name,
                    effect_size=test_result.effect_size,
                    confidence_interval=test_result.confidence_interval
                )
                adjusted_results[group][metric] = adjusted_result
                p_value_idx += 1
        
        return adjusted_results 

    def analyze_metrics(self, metrics_data: Union[Dict, List[Dict]], reference_group: str, test_config: Dict[str, Any]) -> Dict[str, Dict[str, StatTestResult]]:
        """Analyzes metrics for statistical significance against a reference group."""
        default_config = {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "none",
            "bootstrap_iterations": 1000,
            "confidence_level": 0.95,
            "alternative": "two-sided"
        }
        
        config = {**default_config, **test_config}
        self._validate_config(config)
        
        if isinstance(metrics_data, list):
            results = self._analyze_bootstrapped_metrics(metrics_data, reference_group, config)
        else:
            results = self._analyze_single_metrics(metrics_data, reference_group, config)
        
        if config["adjust_method"] != "none":
            results = self._adjust_p_values(results, config["adjust_method"], config["alpha"])
        
        return results

    def analyze_all_group_comparisons(self, metrics_data: Union[Dict, List[Dict]], test_config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, StatTestResult]]]:
        """Performs statistical tests between all possible pairs of groups."""
        default_config = {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "none",
            "bootstrap_iterations": 1000,
            "confidence_level": 0.95,
            "alternative": "two-sided"
        }
        
        config = {**default_config, **test_config}
        self._validate_config(config)
        
        groups = list(metrics_data[0].keys()) if isinstance(metrics_data, list) else list(metrics_data.keys())
        results = {group: {} for group in groups}
        
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                if isinstance(metrics_data, list):
                    pair_results = self._analyze_bootstrapped_metrics_pair(metrics_data, group1, group2, config)
                else:
                    pair_results = self._analyze_single_metrics_pair(metrics_data, group1, group2, config)
                
                results[group1][group2] = pair_results
                results[group2][group1] = pair_results
        
        if config["adjust_method"] != "none":
            results = self._adjust_p_values_all_pairs(results, config["adjust_method"], config["alpha"])
        
        return results

    def _analyze_bootstrapped_metrics_pair(self, bootstrap_metrics: List[Dict], group1: str, group2: str, config: Dict[str, Any]) -> Dict[str, StatTestResult]:
        """Analyzes metrics between two groups in bootstrapped data."""
        results = {}
        test_func = config.get("custom_test_func") or self._test_implementations[config["test_type"]]
        
        group1_metrics = self._reorganize_bootstrap_data(bootstrap_metrics)[group1]
        group2_metrics = self._reorganize_bootstrap_data(bootstrap_metrics)[group2]
        
        for metric_name in group1_metrics.keys():
            if isinstance(group1_metrics[metric_name], list) and len(group1_metrics[metric_name]) > 0:
                test_result = test_func(
                    group1_metrics[metric_name],
                    group2_metrics[metric_name],
                    config
                )
                results[metric_name] = test_result
        
        return results

    def _analyze_single_metrics_pair(self, metrics: Dict, group1: str, group2: str, config: Dict[str, Any]) -> Dict[str, StatTestResult]:
        """Analyzes metrics between two groups in non-bootstrapped data."""
        results = {}
        test_func = config.get("custom_test_func") or self._test_implementations[config["test_type"]]
        
        for metric_name in metrics[group1].keys():
            if isinstance(metrics[group1][metric_name], (int, float, np.ndarray)):
                group1_data = metrics[group1][metric_name]
                group2_data = metrics[group2][metric_name]
                
                if isinstance(group1_data, (int, float)):
                    group1_data = np.array([group1_data])
                if isinstance(group2_data, (int, float)):
                    group2_data = np.array([group2_data])
                
                test_result = test_func(group1_data, group2_data, config)
                results[metric_name] = test_result
        
        return results

    def _adjust_p_values_all_pairs(self, results: Dict[str, Dict[str, Dict[str, StatTestResult]]], method: str, alpha: float) -> Dict[str, Dict[str, Dict[str, StatTestResult]]]:
        """Adjusts p-values for all pairwise comparisons."""
        p_values = []
        for group1_results in results.values():
            for group2_results in group1_results.values():
                for test_result in group2_results.values():
                    p_values.append(test_result.p_value)
        
        if not p_values:
            return results
            
        if method == "bonferroni":
            adjusted_p_values = multipletests(p_values, alpha=alpha, method='bonferroni')[1]
        elif method == "fdr_bh":
            adjusted_p_values = multipletests(p_values, alpha=alpha, method='fdr_bh')[1]
        elif method == "holm":
            adjusted_p_values = multipletests(p_values, alpha=alpha, method='holm')[1]
        else:
            return results
        
        p_value_idx = 0
        adjusted_results = {}
        for group1, group1_results in results.items():
            adjusted_results[group1] = {}
            for group2, group2_results in group1_results.items():
                adjusted_results[group1][group2] = {}
                for metric, test_result in group2_results.items():
                    adjusted_result = StatTestResult(
                        statistic=test_result.statistic,
                        p_value=adjusted_p_values[p_value_idx],
                        is_significant=adjusted_p_values[p_value_idx] < alpha,
                        test_name=test_result.test_name,
                        effect_size=test_result.effect_size,
                        confidence_interval=test_result.confidence_interval
                    )
                    adjusted_results[group1][group2][metric] = adjusted_result
                    p_value_idx += 1
        
        return adjusted_results

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validates test configuration parameters."""
        if config["test_type"] not in self.AVAILABLE_TESTS and not config.get("custom_test_func"):
            raise ValueError(f"Invalid test type. Available tests: {list(self.AVAILABLE_TESTS.keys())}")
        
        if config["adjust_method"] not in self.ADJUSTMENT_METHODS:
            raise ValueError(f"Invalid adjustment method. Available methods: {list(self.ADJUSTMENT_METHODS.keys())}")
        
        if not 0 < config["alpha"] < 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        if not 0 < config["confidence_level"] < 1:
            raise ValueError("Confidence level must be between 0 and 1")

    def _analyze_single_metrics(self, metrics: Dict, reference_group: str, config: Dict[str, Any]) -> Dict[str, Dict[str, StatTestResult]]:
        """Analyzes non-bootstrapped metrics against a reference group."""
        results = {}
        test_func = config.get("custom_test_func") or self._test_implementations[config["test_type"]]
        
        ref_metrics = metrics[reference_group] 
        for group, group_metrics in metrics.items():
            if group == reference_group:
                continue
                
            results[group] = {}
            for metric_name, value in group_metrics.items():
                if isinstance(value, (int, float)):
                    test_result = test_func(
                        ref_metrics[metric_name],
                        value,
                        config
                    )
                    results[group][metric_name] = test_result
        
        return results

    def _analyze_bootstrapped_metrics(self, bootstrap_metrics: List[Dict], reference_group: str, config: Dict[str, Any]) -> Dict[str, Dict[str, StatTestResult]]:
        """Analyzes bootstrapped metrics against a reference group."""
        results = {}
        metrics_by_group = self._reorganize_bootstrap_data(bootstrap_metrics)
        
        ref_distributions = metrics_by_group[reference_group]
        for group, group_distributions in metrics_by_group.items():
            if group == reference_group:
                continue
                
            results[group] = {}
            for metric_name in group_distributions.keys():
                test_result = self._bootstrap_test(
                    ref_distributions[metric_name],
                    group_distributions[metric_name],
                    config
                )
                results[group][metric_name] = test_result
        
        return results

    def _reorganize_bootstrap_data(self, bootstrap_metrics: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
        """Reorganizes bootstrapped data into distributions by group and metric."""
        organized_data = {}
        
        sample = bootstrap_metrics[0]
        for group in sample.keys():
            organized_data[group] = {}
            for metric in sample[group].keys():
                if isinstance(sample[group][metric], (int, float, np.ndarray)):
                    organized_data[group][metric] = []
        
        for bootstrap_sample in bootstrap_metrics:
            for group in bootstrap_sample.keys():
                for metric, value in bootstrap_sample[group].items():
                    if isinstance(value, (int, float, np.ndarray)):
                        if isinstance(value, (int, float)):
                            value = np.array([value])
                        organized_data[group][metric].extend(value)
        
        return organized_data

    def _bootstrap_test(self, ref_distribution: List[float], comp_distribution: List[float], config: Dict[str, Any]) -> StatTestResult:
        """Performs bootstrap test using scipy.stats.bootstrap."""
        from scipy.stats import bootstrap
        
        def statistic(x, y):
            return np.mean(y) - np.mean(x)
            
        # Convert to numpy arrays
        ref_data = np.array(ref_distribution)
        comp_data = np.array(comp_distribution)
        
        # Perform bootstrap
        bootstrap_result = bootstrap(
            data=(ref_data, comp_data),
            statistic=statistic,
            n_resamples=config["bootstrap_iterations"],
            confidence_level=config["confidence_level"],
            method='percentile'
        )
        
        # Calculate p-value
        mean_diff = statistic(ref_data, comp_data)
        p_value = np.mean(np.abs(bootstrap_result.bootstrap_distribution) >= np.abs(mean_diff))
        
        # Get confidence interval
        ci_lower, ci_upper = bootstrap_result.confidence_interval
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(ref_distribution, comp_distribution)
        
        return StatTestResult(
            statistic=mean_diff,
            p_value=p_value,
            is_significant=p_value < config["alpha"],
            test_name="Bootstrap Test",
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper)
        )

    def _wilcoxon_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any]) -> StatTestResult:
        """Performs Wilcoxon signed-rank test for paired samples."""
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
            
        statistic, p_value = stats.wilcoxon(
            ref_data,
            comp_data,
            alternative=config.get("alternative", "two-sided")
        )
        
        effect_size = self._calculate_effect_size(ref_data, comp_data)
        
        return StatTestResult(
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < config["alpha"],
            test_name="Wilcoxon Signed-Rank Test",
            effect_size=effect_size
        )

    def _permutation_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any]) -> StatTestResult:
        """Performs custom permutation test with user-defined test statistic."""
        test_stat_func = config.get("test_statistic", lambda x, y: np.mean(x) - np.mean(y))
        
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
        
        observed_stat = test_stat_func(ref_data, comp_data)
        
        combined = np.concatenate([ref_data, comp_data])
        n_ref = len(ref_data)
        n_iterations = config["bootstrap_iterations"]
        
        perm_stats = []
        for _ in range(n_iterations):
            np.random.shuffle(combined)
            perm_ref = combined[:n_ref]
            perm_comp = combined[n_ref:]
            perm_stats.append(test_stat_func(perm_ref, perm_comp))
        
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        
        effect_size = self._calculate_effect_size(ref_data, comp_data)
        
        return StatTestResult(
            statistic=observed_stat,
            p_value=p_value,
            is_significant=p_value < config["alpha"],
            test_name="Custom Permutation Test",
            effect_size=effect_size
        ) 