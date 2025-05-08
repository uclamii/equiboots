import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import pingouin as pg


@dataclass
class StatTestResult:
    """Stores statistical test results including test statistic, p-value, and significance."""
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    is_significant: bool
    test_name: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        # Ensure is_significant is a Python bool
        self.is_significant = bool(self.is_significant)

class StatisticalTester:
    """Performs statistical significance testing on metrics with support for various tests and data types."""
    
    AVAILABLE_TESTS = {
        "mann_whitney": "Mann-Whitney U test (non-parametric)",
        "t_test": "Welch's t-test (parametric)",
        "ks_test": "Kolmogorov-Smirnov test",
        "permutation": "Permutation test (non-parametric)",
        "wilcoxon": "Wilcoxon signed-rank test",
        "chi_square": "Chi-square test",
        "z_test": "Z-test (parametric)",
    }
    
    ADJUSTMENT_METHODS = {
        "bonferroni": "Bonferroni correction",
        "fdr_bh": "Benjamini-Hochberg FDR",
        "holm": "Holm-Bonferroni",
        "none": "No correction"
    }

    def __init__(self, critical_value):
        """Initializes StatisticalTester with default test implementations."""
        self._test_implementations = {
            "mann_whitney": self._mann_whitney_test,
            "t_test": self._t_test,
            "ks_test": self._ks_test,
            "permutation": self._permutation_test,
            "wilcoxon": self._wilcoxon_test,
            "chi_square": self._chi_square_test,
            "z_test": self._z_test, 
        }
        self.critical_value: Optional[float] = None

    def _mann_whitney_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any],) -> StatTestResult:
        """Performs Mann-Whitney U test using pingouin for non-parametric comparison.
        
        Args:
            ref_data: List of values from reference group
            comp_data: List of values from comparison group
            config: Configuration dictionary containing test parameters
            
        Returns:
            StatTestResult object containing test results
        """
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
            
        # Convert to numpy arrays
        ref_array = np.array(ref_data)
        comp_array = np.array(comp_data)
        
        # Use pingouin's implementation
        result = pg.mwu(
            ref_array,
            comp_array,
            alternative=config.get("alternative", "two-sided")
        )
        
        return StatTestResult(
            statistic=result['U-val'].iloc[0],
            p_value=result['p-val'].iloc[0],
            is_significant=result['p-val'].iloc[0] < config.get("alpha", 0.05),
            test_name="Mann-Whitney U",
            effect_size=result['RBC'].iloc[0]  # Rank-biserial correlation as effect size
        )

    def _t_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any]) -> StatTestResult:
        """Performs Welch's t-test using pingouin for parametric comparison.
        
        Args:
            ref_data: List of values from reference group
            comp_data: List of values from comparison group
            config: Configuration dictionary containing test parameters
            
        Returns:
            StatTestResult object containing test results
        """
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
            
        # Convert to numpy arrays
        ref_array = np.array(ref_data)
        comp_array = np.array(comp_data)
        
        # Use pingouin's implementation
        result = pg.ttest(
            ref_array,
            comp_array,
            paired=False,
            alternative=config.get("alternative", "two-sided")
        )
        
        return StatTestResult(
            statistic=result['T'].iloc[0],
            p_value=result['p-val'].iloc[0],
            is_significant=result['p-val'].iloc[0] < config.get("alpha", 0.05),
            test_name="Welch's t-test",
            effect_size=result['cohen-d'].iloc[0],
            confidence_interval=(result['CI95%'].iloc[0][0], result['CI95%'].iloc[0][1])
        )

    def _ks_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any]) -> StatTestResult:
        """Performs Kolmogorov-Smirnov test using scipy for distribution comparison.
        
        Args:
            ref_data: List of values from reference group
            comp_data: List of values from comparison group
            config: Configuration dictionary containing test parameters
            
        Returns:
            StatTestResult object containing test results
        """
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
            
        # Convert to numpy arrays
        ref_array = np.array(ref_data)
        comp_array = np.array(comp_data)
        
        # Use scipy's implementation
        statistic, p_value = stats.ks_2samp(ref_array, comp_array)
        
        effect_size = self._calculate_effect_size(ref_array, comp_array)
        
        return StatTestResult(
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < config.get("alpha", 0.05),
            test_name="Kolmogorov-Smirnov",
            effect_size=effect_size
        )
    
    def _permutation_test(self, ref_distribution: List[float], comp_distribution: List[float], config: Dict[str, Any]) -> StatTestResult:
        """Performs permutation test with confidence intervals.
        
        Args:
            ref_distribution: List of values from reference group
            comp_distribution: List of values from comparison group
            config: Configuration dictionary containing test parameters
            
        Returns:
            StatTestResult object containing test results
        """
        # Convert to numpy arrays
        ref_array = np.array(ref_distribution)
        comp_array = np.array(comp_distribution)
        
        # Define the test statistic (mean difference)
        def statistic(x, y):
            return np.mean(y) - np.mean(x)
        
        # Perform permutation test
        result = stats.permutation_test(
            (ref_array, comp_array),
            statistic,
            n_resamples=config.get("bootstrap_iterations", 1000),
            alternative=config.get("alternative", "two-sided"),
            random_state=42  # For reproducibility
        )
        
        # Calculate confidence intervals using bootstrap
        ci_level = config.get("confidence_level", 0.95)
        ci_lower = np.percentile(result.null_distribution, (1 - ci_level) * 100 / 2)
        ci_upper = np.percentile(result.null_distribution, (1 + ci_level) * 100 / 2)
        
        effect_size = self._calculate_effect_size(ref_distribution, comp_distribution)
        
        return StatTestResult(
            statistic=result.statistic,
            p_value=result.pvalue,
            is_significant=result.pvalue < config.get("alpha", 0.05),
            test_name="Permutation Test",
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper)
        )

    def _wilcoxon_test(self, ref_data: Union[float, List[float]], comp_data: Union[float, List[float]], config: Dict[str, Any]) -> StatTestResult:
        """Performs Wilcoxon signed-rank test using pingouin for paired samples.
        
        Args:
            ref_data: List of values from reference group
            comp_data: List of values from comparison group
            config: Configuration dictionary containing test parameters
            
        Returns:
            StatTestResult object containing test results
        """
        if isinstance(ref_data, (int, float)):
            ref_data = [ref_data]
        if isinstance(comp_data, (int, float)):
            comp_data = [comp_data]
            
        # Convert to numpy arrays
        ref_array = np.array(ref_data)
        comp_array = np.array(comp_data)
        
        # Use pingouin's implementation
        result = pg.wilcoxon(
            ref_array,
            comp_array,
            alternative=config.get("alternative", "two-sided")
        )
        
        return StatTestResult(
            statistic=result['W-val'].iloc[0],
            p_value=result['p-val'].iloc[0],
            is_significant=result['p-val'].iloc[0] < config.get("alpha", 0.05),
            test_name="Wilcoxon Signed-Rank Test",
            effect_size=result['RBC'].iloc[0]  # Rank-biserial correlation as effect size
        ) 
    
    def _chi_square_test(self, ref_data: Union[int, List[int]], comp_data: Union[int, List[int]], config: Dict[str, Any]) -> StatTestResult:
        """Performs Chi-square test for categorical data.
        
        Args:
            ref_data: List of values from reference group
            comp_data: List of values from comparison group
            config: Configuration dictionary containing test parameters
            
        Returns:
            StatTestResult object containing test results
        """
        # Convert to numpy arrays
        ref_array = np.array(ref_data)
        comp_array = np.array(comp_data)
        
        # Create contingency table
        contingency_table = pd.crosstab(ref_array, comp_array)
        
        # Use scipy's implementation
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        return StatTestResult(
            statistic=chi2,
            p_value=p_value,
            is_significant=p_value < config.get("alpha", 0.05),
            test_name="Chi-Square Test"
        )
      

    def _calculate_effect_size(self, ref_data: List[float], comp_data: List[float]) -> float:
        """Calculates Cohen's d effect size using pingouin.
        
        Args:
            ref_data: List of values from reference group
            comp_data: List of values from comparison group
            
        Returns:
            float: Cohen's d effect size
        """
        # Convert lists to numpy arrays
        ref_array = np.array(ref_data)
        comp_array = np.array(comp_data)
        
        # Calculate Cohen's d using pingouin
        effect_size = pg.compute_effsize(ref_array, comp_array, eftype='cohen')
        return effect_size

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
                test_result = self._permutation_test(
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
                            organized_data[group][metric].append(float(value))
                        else:
                            organized_data[group][metric].extend(value)
        
        return organized_data

    