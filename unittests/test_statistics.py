import numpy as np
import pandas as pd
import pytest
from equiboots import EquiBoots
from equiboots.StatisticalTester import StatisticalTester, StatTestResult

def test_statistical_tester_initialization():
    """Test initialization of StatisticalTester class"""
    tester = StatisticalTester()
    assert isinstance(tester, StatisticalTester)
    assert hasattr(tester, '_test_implementations')
    assert 'mann_whitney' in tester._test_implementations
    assert 't_test' in tester._test_implementations
    assert 'ks_test' in tester._test_implementations
    assert 'bootstrap_test' in tester._test_implementations
    assert 'wilcoxon' in tester._test_implementations
    assert 'permutation' in tester._test_implementations

def test_stat_test_result_dataclass():
    """Test StatTestResult dataclass"""
    result = StatTestResult(
        statistic=1.5,
        p_value=0.05,
        is_significant=True,
        test_name="test",
        effect_size=0.8,
        confidence_interval=(0.5, 2.5)
    )
    assert result.statistic == 1.5
    assert result.p_value == 0.05
    assert result.is_significant is True
    assert result.test_name == "test"
    assert result.effect_size == 0.8
    assert result.confidence_interval == (0.5, 2.5)

def test_mann_whitney_test():
    """Test Mann-Whitney U test implementation"""
    tester = StatisticalTester()
    ref_data = np.random.normal(0, 1, 100)
    comp_data = np.random.normal(1, 1, 100)
    
    result = tester._mann_whitney_test(ref_data, comp_data, {"alpha": 0.05})
    assert isinstance(result, StatTestResult)
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'is_significant')
    assert hasattr(result, 'effect_size')

def test_t_test():
    """Test Welch's t-test implementation"""
    tester = StatisticalTester()
    ref_data = np.random.normal(0, 1, 100)
    comp_data = np.random.normal(1, 1, 100)
    
    result = tester._t_test(ref_data, comp_data, {"alpha": 0.05})
    assert isinstance(result, StatTestResult)
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'is_significant')
    assert hasattr(result, 'effect_size')

def test_ks_test():
    """Test Kolmogorov-Smirnov test implementation"""
    tester = StatisticalTester()
    ref_data = np.random.normal(0, 1, 100)
    comp_data = np.random.normal(1, 1, 100)
    
    result = tester._ks_test(ref_data, comp_data, {"alpha": 0.05})
    assert isinstance(result, StatTestResult)
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'is_significant')
    assert hasattr(result, 'effect_size')

def test_bootstrap_test():
    """Test bootstrap test implementation"""
    tester = StatisticalTester()
    ref_data = np.random.normal(0, 1, 100)
    comp_data = np.random.normal(1, 1, 100)
    
    result = tester._bootstrap_test(ref_data, comp_data, {
        "alpha": 0.05,
        "bootstrap_iterations": 1000,
        "confidence_level": 0.95
    })
    assert isinstance(result, StatTestResult)
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'is_significant')
    assert hasattr(result, 'effect_size')
    assert hasattr(result, 'confidence_interval')

def test_wilcoxon_test():
    """Test Wilcoxon signed-rank test implementation"""
    tester = StatisticalTester()
    ref_data = np.random.normal(0, 1, 100)
    comp_data = np.random.normal(1, 1, 100)
    
    result = tester._wilcoxon_test(ref_data, comp_data, {"alpha": 0.05})
    assert isinstance(result, StatTestResult)
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'is_significant')
    assert hasattr(result, 'effect_size')

def test_permutation_test():
    """Test permutation test implementation"""
    tester = StatisticalTester()
    ref_data = np.random.normal(0, 1, 100)
    comp_data = np.random.normal(1, 1, 100)
    
    result = tester._permutation_test(ref_data, comp_data, {
        "alpha": 0.05,
        "bootstrap_iterations": 1000,
        "test_statistic": lambda x, y: np.mean(x) - np.mean(y)
    })
    assert isinstance(result, StatTestResult)
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'is_significant')
    assert hasattr(result, 'effect_size')

def test_adjust_p_values():
    """Test p-value adjustment methods"""
    tester = StatisticalTester()
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    # Test Bonferroni correction
    bonferroni_results = tester._adjust_p_values(
        {"group1": {"metric1": StatTestResult(0, p_values[0], True, "test")}},
        "bonferroni",
        0.05
    )
    assert isinstance(bonferroni_results, dict)
    
    # Test FDR BH correction
    fdr_results = tester._adjust_p_values(
        {"group1": {"metric1": StatTestResult(0, p_values[0], True, "test")}},
        "fdr_bh",
        0.05
    )
    assert isinstance(fdr_results, dict)
    
    # Test Holm correction
    holm_results = tester._adjust_p_values(
        {"group1": {"metric1": StatTestResult(0, p_values[0], True, "test")}},
        "holm",
        0.05
    )
    assert isinstance(holm_results, dict)

def test_analyze_metrics():
    """Test analyze_metrics method"""
    tester = StatisticalTester()
    metrics_data = {
        "group1": {"metric1": 0.8, "metric2": 0.6},
        "group2": {"metric1": 0.9, "metric2": 0.7}
    }
    
    results = tester.analyze_metrics(
        metrics_data,
        "group1",
        {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "none"
        }
    )
    assert isinstance(results, dict)
    assert "group2" in results
    assert "metric1" in results["group2"]
    assert "metric2" in results["group2"]

def test_equiboots_statistical_analysis():
    """Test statistical analysis through EquiBoots class"""
    # Generate synthetic data with positive values
    np.random.seed(42)
    n_samples = 100
    # Generate positive values by using exponential of normal distribution
    y_true = np.exp(np.random.normal(0, 0.5, n_samples))
    y_pred = y_true * (1 + np.random.normal(0, 0.1, n_samples))
    fairness_df = pd.DataFrame({
        'race': np.random.choice(['white', 'black', 'asian'], n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples)
    })
    
    # Initialize EquiBoots
    eq = EquiBoots(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=None,  # Add y_prob=None for regression task
        fairness_df=fairness_df,
        fairness_vars=['race', 'sex'],
        reference_groups=['white', 'M'],
        task='regression',
        bootstrap_flag=False
    )
    
    # Test list_available_tests
    tests = eq.list_available_tests()
    assert isinstance(tests, dict)
    assert 'mann_whitney' in tests
    assert 't_test' in tests
    
    # Test list_adjustment_methods
    methods = eq.list_adjustment_methods()
    assert isinstance(methods, dict)
    assert 'bonferroni' in methods
    assert 'fdr_bh' in methods
    
    # Test analyze_statistical_significance
    eq.grouper(groupings_vars=['race'])
    race_data = eq.slicer('race')
    race_metrics = eq.get_metrics(race_data)
    
    results = eq.analyze_statistical_significance(
        metric_dict=race_metrics,
        var_name='race',
        test_config={
            'test_type': 'mann_whitney',
            'alpha': 0.05,
            'adjust_method': 'none'
        }
    )
    assert isinstance(results, dict)
    assert 'black' in results
    assert 'asian' in results

def test_analyze_all_group_comparisons():
    """Test the analyze_all_group_comparisons method with synthetic data."""
    # Create synthetic data
    n_samples = 100
    metrics_data = {
        "group1": {
            "metric1": np.random.normal(0, 1, n_samples),
            "metric2": np.random.normal(1, 1, n_samples)
        },
        "group2": {
            "metric1": np.random.normal(0.5, 1, n_samples),
            "metric2": np.random.normal(1.5, 1, n_samples)
        },
        "group3": {
            "metric1": np.random.normal(1, 1, n_samples),
            "metric2": np.random.normal(2, 1, n_samples)
        }
    }
    
    # Initialize tester
    tester = StatisticalTester()
    
    # Test with different configurations
    test_configs = [
        {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "none"
        },
        {
            "test_type": "t_test",
            "alpha": 0.01,
            "adjust_method": "bonferroni"
        },
        {
            "test_type": "bootstrap_test",
            "alpha": 0.05,
            "adjust_method": "fdr_bh",
            "bootstrap_iterations": 100
        }
    ]
    
    for config in test_configs:
        results = tester.analyze_all_group_comparisons(metrics_data, config)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert all(isinstance(group_results, dict) for group_results in results.values())
        
        # Check that all group pairs are compared
        groups = list(metrics_data.keys())
        for group1 in groups:
            for group2 in groups:
                if group1 != group2:
                    assert group2 in results[group1]
                    assert isinstance(results[group1][group2], dict)
                    
                    # Check that all metrics are compared
                    for metric in metrics_data[group1].keys():
                        assert metric in results[group1][group2]
                        result = results[group1][group2][metric]
                        assert isinstance(result, StatTestResult)
                        assert hasattr(result, 'statistic')
                        assert hasattr(result, 'p_value')
                        assert hasattr(result, 'is_significant')
                        assert hasattr(result, 'test_name')
                        assert hasattr(result, 'effect_size')

def test_all_group_comparisons_with_bootstrapped_data():
    """Test all-group comparisons with bootstrapped data."""
    # Create synthetic bootstrapped data
    n_bootstraps = 10
    n_samples = 100
    bootstrap_metrics = []
    
    for _ in range(n_bootstraps):
        metrics = {
            "group1": {
                "metric1": np.random.normal(0, 1, n_samples),
                "metric2": np.random.normal(1, 1, n_samples)
            },
            "group2": {
                "metric1": np.random.normal(0.5, 1, n_samples),
                "metric2": np.random.normal(1.5, 1, n_samples)
            },
            "group3": {
                "metric1": np.random.normal(1, 1, n_samples),
                "metric2": np.random.normal(2, 1, n_samples)
            }
        }
        bootstrap_metrics.append(metrics)
    
    # Initialize tester
    tester = StatisticalTester()
    
    # Test configuration
    config = {
        "test_type": "bootstrap_test",
        "alpha": 0.05,
        "adjust_method": "none",
        "bootstrap_iterations": 100
    }
    
    results = tester.analyze_all_group_comparisons(bootstrap_metrics, config)
    
    # Verify results structure
    assert isinstance(results, dict)
    assert all(isinstance(group_results, dict) for group_results in results.values())
    
    # Check that all group pairs are compared
    groups = list(bootstrap_metrics[0].keys())
    for group1 in groups:
        for group2 in groups:
            if group1 != group2:
                assert group2 in results[group1]
                assert isinstance(results[group1][group2], dict)
                
                # Check that all metrics are compared
                for metric in bootstrap_metrics[0][group1].keys():
                    assert metric in results[group1][group2]
                    result = results[group1][group2][metric]
                    assert isinstance(result, StatTestResult)
                    assert hasattr(result, 'statistic')
                    assert hasattr(result, 'p_value')
                    assert hasattr(result, 'is_significant')
                    assert hasattr(result, 'test_name')
                    assert hasattr(result, 'effect_size')
                    assert hasattr(result, 'confidence_interval')

def test_all_group_comparisons_with_different_metrics():
    """Test all-group comparisons with different types of metrics."""
    # Create synthetic data with different metric types
    n_samples = 100
    metrics_data = {
        "group1": {
            "continuous": np.random.normal(0, 1, n_samples),
            "binary": np.random.binomial(1, 0.5, n_samples),
            "count": np.random.poisson(5, n_samples)
        },
        "group2": {
            "continuous": np.random.normal(0.5, 1, n_samples),
            "binary": np.random.binomial(1, 0.6, n_samples),
            "count": np.random.poisson(6, n_samples)
        },
        "group3": {
            "continuous": np.random.normal(1, 1, n_samples),
            "binary": np.random.binomial(1, 0.7, n_samples),
            "count": np.random.poisson(7, n_samples)
        }
    }
    
    # Initialize tester
    tester = StatisticalTester()
    
    # Test with different test types
    test_configs = [
        {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "none"
        },
        {
            "test_type": "t_test",
            "alpha": 0.05,
            "adjust_method": "none"
        },
        {
            "test_type": "bootstrap_test",
            "alpha": 0.05,
            "adjust_method": "none",
            "bootstrap_iterations": 100
        }
    ]
    
    for config in test_configs:
        results = tester.analyze_all_group_comparisons(metrics_data, config)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert all(isinstance(group_results, dict) for group_results in results.values())
        
        # Check that all group pairs are compared
        groups = list(metrics_data.keys())
        for group1 in groups:
            for group2 in groups:
                if group1 != group2:
                    assert group2 in results[group1]
                    assert isinstance(results[group1][group2], dict)
                    
                    # Check that all metrics are compared
                    for metric in metrics_data[group1].keys():
                        assert metric in results[group1][group2]
                        result = results[group1][group2][metric]
                        assert isinstance(result, StatTestResult)
                        assert hasattr(result, 'statistic')
                        assert hasattr(result, 'p_value')
                        assert hasattr(result, 'is_significant')
                        assert hasattr(result, 'test_name')
                        assert hasattr(result, 'effect_size')

def test_all_group_comparisons_with_p_value_adjustment():
    """Test all-group comparisons with different p-value adjustment methods."""
    # Create synthetic data
    n_samples = 100
    metrics_data = {
        "group1": {
            "metric1": np.random.normal(0, 1, n_samples),
            "metric2": np.random.normal(1, 1, n_samples)
        },
        "group2": {
            "metric1": np.random.normal(0.5, 1, n_samples),
            "metric2": np.random.normal(1.5, 1, n_samples)
        },
        "group3": {
            "metric1": np.random.normal(1, 1, n_samples),
            "metric2": np.random.normal(2, 1, n_samples)
        }
    }
    
    # Initialize tester
    tester = StatisticalTester()
    
    # Test different adjustment methods
    adjustment_methods = ["bonferroni", "fdr_bh", "holm"]
    
    for method in adjustment_methods:
        config = {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": method
        }
        
        results = tester.analyze_all_group_comparisons(metrics_data, config)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert all(isinstance(group_results, dict) for group_results in results.values())
        
        # Check that p-values are adjusted
        p_values = []
        for group1_results in results.values():
            for group2_results in group1_results.values():
                for test_result in group2_results.values():
                    p_values.append(test_result.p_value)
        
        # Verify that p-values are within valid range
        assert all(0 <= p <= 1 for p in p_values)
        
        # For bonferroni and holm, verify that p-values are not smaller than original
        if method in ["bonferroni", "holm"]:
            original_results = tester.analyze_all_group_comparisons(
                metrics_data,
                {**config, "adjust_method": "none"}
            )
            original_p_values = []
            for group1_results in original_results.values():
                for group2_results in group1_results.values():
                    for test_result in group2_results.values():
                        original_p_values.append(test_result.p_value)
            
            assert all(adj_p >= orig_p for adj_p, orig_p in zip(p_values, original_p_values))

def test_equiboots_all_group_comparisons():
    """Test all-group comparisons through the EquiBoots class."""
    # Generate synthetic data with positive values
    n_samples = 100
    y_true = np.exp(np.random.normal(0, 0.5, n_samples))  # Ensure positive values
    y_pred = y_true * (1 + np.random.normal(0, 0.1, n_samples))  # Add small noise
    
    # Create demographic data
    groups = np.random.choice(["A", "B", "C"], n_samples)
    fairness_df = pd.DataFrame({"group": groups})
    
    # Initialize EquiBoots
    eq = EquiBoots(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=None,
        fairness_df=fairness_df,
        fairness_vars=["group"],
        reference_groups=["A"],
        task="regression",
        bootstrap_flag=True
    )
    
    # Get metrics
    eq.grouper(groupings_vars=["group"])
    group_data = eq.slicer("group")
    group_metrics = eq.get_metrics(group_data)
    
    # Test different configurations
    test_configs = [
        None,  # Default settings
        {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "bonferroni"
        },
        {
            "test_type": "bootstrap_test",
            "alpha": 0.01,
            "adjust_method": "fdr_bh",
            "bootstrap_iterations": 100
        }
    ]
    
    for config in test_configs:
        # Perform statistical testing
        results = eq.analyze_statistical_significance(
            metric_dict=group_metrics,
            var_name="group",
            test_config=config
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert all(isinstance(group_results, dict) for group_results in results.values())
        
        # Check that all groups are compared
        groups = list(group_metrics[0].keys())
        for group in groups:
            if group != "A":  # Skip reference group
                assert group in results
                assert isinstance(results[group], dict)
                
                # Check that all metrics are compared
                for metric in group_metrics[0][group].keys():
                    if isinstance(group_metrics[0][group][metric], (int, float)):
                        assert metric in results[group]
                        result = results[group][metric]
                        assert isinstance(result, StatTestResult)
                        assert hasattr(result, 'statistic')
                        assert hasattr(result, 'p_value')
                        assert hasattr(result, 'is_significant')
                        assert hasattr(result, 'test_name')
                        assert hasattr(result, 'effect_size') 