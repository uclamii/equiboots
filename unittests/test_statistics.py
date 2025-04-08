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
    assert 'permutation' in tester._test_implementations
    assert 'wilcoxon' in tester._test_implementations

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

def test_permutation_test():
    """Test permutation test implementation"""
    tester = StatisticalTester()
    ref_data = np.random.normal(0, 1, 100)
    comp_data = np.random.normal(1, 1, 100)
    
    result = tester._permutation_test(ref_data, comp_data, {
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
    """Test all-group comparisons functionality."""
    # Create synthetic data
    n_samples = 100
    groups = ["A", "B", "C"]
    metrics = {
        "A": {
            "metric1": np.random.normal(0, 1, n_samples),
            "metric2": np.random.normal(0, 1, n_samples)
        },
        "B": {
            "metric1": np.random.normal(0.5, 1, n_samples),
            "metric2": np.random.normal(0.5, 1, n_samples)
        },
        "C": {
            "metric1": np.random.normal(1, 1, n_samples),
            "metric2": np.random.normal(1, 1, n_samples)
        }
    }
    
    # Test with different configurations
    test_configs = [
        {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "bonferroni"
        },
        {
            "test_type": "t_test",
            "alpha": 0.05,
            "adjust_method": "holm"
        },
        {
            "test_type": "ks_test",
            "alpha": 0.05,
            "adjust_method": "fdr_bh"
        },
        {
            "test_type": "permutation",
            "alpha": 0.05,
            "adjust_method": "none",
            "bootstrap_iterations": 1000
        },
        {
            "test_type": "wilcoxon",
            "alpha": 0.05,
            "adjust_method": "bonferroni"
        }
    ]
    
    for config in test_configs:
        tester = StatisticalTester()
        results = tester.analyze_all_group_comparisons(metrics, config)
        
        # Check that all groups are present in results
        assert set(results.keys()) == set(groups)
        
        # Check that all group pairs are compared
        for group1 in groups:
            assert set(results[group1].keys()) == set(groups) - {group1}
            for group2 in results[group1].keys():
                # Check that results contain all metrics
                assert set(results[group1][group2].keys()) == {"metric1", "metric2"}
                
                # Check that each metric has a valid StatTestResult
                for metric, result in results[group1][group2].items():
                    assert isinstance(result, StatTestResult)
                    assert isinstance(result.statistic, float)
                    assert isinstance(result.p_value, float)
                    assert isinstance(result.is_significant, bool)
                    assert isinstance(result.test_name, str)
                    assert isinstance(result.effect_size, float)
                    if result.confidence_interval:
                        assert isinstance(result.confidence_interval, tuple)
                        assert len(result.confidence_interval) == 2
                        assert all(isinstance(x, float) for x in result.confidence_interval)

def test_all_group_comparisons_with_bootstrapped_data():
    """Test all-group comparisons with bootstrapped data."""
    # Create synthetic bootstrapped data
    n_bootstraps = 50
    n_samples = 100
    groups = ["A", "B", "C"]
    metrics = []
    
    for _ in range(n_bootstraps):
        bootstrap_sample = {
            "A": {
                "metric1": np.random.normal(0, 1, n_samples),
                "metric2": np.random.normal(0, 1, n_samples)
            },
            "B": {
                "metric1": np.random.normal(0.5, 1, n_samples),
                "metric2": np.random.normal(0.5, 1, n_samples)
            },
            "C": {
                "metric1": np.random.normal(1, 1, n_samples),
                "metric2": np.random.normal(1, 1, n_samples)
            }
        }
        metrics.append(bootstrap_sample)
    
    # Test with different configurations
    test_configs = [
        {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "bonferroni"
        },
        {
            "test_type": "t_test",
            "alpha": 0.05,
            "adjust_method": "holm"
        },
        {
            "test_type": "ks_test",
            "alpha": 0.05,
            "adjust_method": "fdr_bh"
        },
        {
            "test_type": "permutation",
            "alpha": 0.05,
            "adjust_method": "none",
            "bootstrap_iterations": 1000
        }
    ]
    
    for config in test_configs:
        tester = StatisticalTester()
        results = tester.analyze_all_group_comparisons(metrics, config)
        
        # Check that all groups are present in results
        assert set(results.keys()) == set(groups)
        
        # Check that all group pairs are compared
        for group1 in groups:
            assert set(results[group1].keys()) == set(groups) - {group1}
            for group2 in results[group1].keys():
                # Check that results contain all metrics
                assert set(results[group1][group2].keys()) == {"metric1", "metric2"}
                
                # Check that each metric has a valid StatTestResult
                for metric, result in results[group1][group2].items():
                    assert isinstance(result, StatTestResult)
                    assert isinstance(result.statistic, float)
                    assert isinstance(result.p_value, float)
                    assert isinstance(result.is_significant, bool)
                    assert isinstance(result.test_name, str)
                    assert isinstance(result.effect_size, float)
                    if result.confidence_interval:
                        assert isinstance(result.confidence_interval, tuple)
                        assert len(result.confidence_interval) == 2
                        assert all(isinstance(x, float) for x in result.confidence_interval)

def test_all_group_comparisons_with_different_metrics():
    """Test all-group comparisons with different types of metrics."""
    # Create synthetic data with different metric types
    n_samples = 100
    groups = ["A", "B", "C"]
    metrics = {
        "A": {
            "continuous": np.random.normal(0, 1, n_samples),
            "binary": np.random.binomial(1, 0.5, n_samples),
            "count": np.random.poisson(5, n_samples)
        },
        "B": {
            "continuous": np.random.normal(0.5, 1, n_samples),
            "binary": np.random.binomial(1, 0.6, n_samples),
            "count": np.random.poisson(6, n_samples)
        },
        "C": {
            "continuous": np.random.normal(1, 1, n_samples),
            "binary": np.random.binomial(1, 0.7, n_samples),
            "count": np.random.poisson(7, n_samples)
        }
    }
    
    # Test with different configurations
    test_configs = [
        {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "bonferroni"
        },
        {
            "test_type": "t_test",
            "alpha": 0.05,
            "adjust_method": "holm"
        },
        {
            "test_type": "ks_test",
            "alpha": 0.05,
            "adjust_method": "fdr_bh"
        },
        {
            "test_type": "permutation",
            "alpha": 0.05,
            "adjust_method": "none",
            "bootstrap_iterations": 1000
        }
    ]
    
    for config in test_configs:
        tester = StatisticalTester()
        results = tester.analyze_all_group_comparisons(metrics, config)
        
        # Check that all groups are present in results
        assert set(results.keys()) == set(groups)
        
        # Check that all group pairs are compared
        for group1 in groups:
            assert set(results[group1].keys()) == set(groups) - {group1}
            for group2 in results[group1].keys():
                # Check that results contain all metrics
                assert set(results[group1][group2].keys()) == {"continuous", "binary", "count"}
                
                # Check that each metric has a valid StatTestResult
                for metric, result in results[group1][group2].items():
                    assert isinstance(result, StatTestResult)
                    assert isinstance(result.statistic, float)
                    assert isinstance(result.p_value, float)
                    assert isinstance(result.is_significant, bool)
                    assert isinstance(result.test_name, str)
                    assert isinstance(result.effect_size, float)
                    if result.confidence_interval:
                        assert isinstance(result.confidence_interval, tuple)
                        assert len(result.confidence_interval) == 2
                        assert all(isinstance(x, float) for x in result.confidence_interval)

def test_all_group_comparisons_with_p_value_adjustment():
    """Test all-group comparisons with different p-value adjustment methods."""
    # Create synthetic data
    n_samples = 100
    groups = ["A", "B", "C"]
    metrics = {
        "A": {
            "metric1": np.random.normal(0, 1, n_samples),
            "metric2": np.random.normal(0, 1, n_samples)
        },
        "B": {
            "metric1": np.random.normal(0.5, 1, n_samples),
            "metric2": np.random.normal(0.5, 1, n_samples)
        },
        "C": {
            "metric1": np.random.normal(1, 1, n_samples),
            "metric2": np.random.normal(1, 1, n_samples)
        }
    }
    
    # Test with different adjustment methods
    adjustment_methods = ["bonferroni", "fdr_bh", "holm", "none"]
    
    for method in adjustment_methods:
        config = {
            "test_type": "t_test",
            "alpha": 0.05,
            "adjust_method": method
        }
        
        tester = StatisticalTester()
        results = tester.analyze_all_group_comparisons(metrics, config)
        
        # Check that all groups are present in results
        assert set(results.keys()) == set(groups)
        
        # Check that all group pairs are compared
        for group1 in groups:
            assert set(results[group1].keys()) == set(groups) - {group1}
            for group2 in results[group1].keys():
                # Check that results contain all metrics
                assert set(results[group1][group2].keys()) == {"metric1", "metric2"}
                
                # Check that each metric has a valid StatTestResult
                for metric, result in results[group1][group2].items():
                    assert isinstance(result, StatTestResult)
                    assert isinstance(result.statistic, float)
                    assert isinstance(result.p_value, float)
                    assert isinstance(result.is_significant, bool)
                    assert isinstance(result.test_name, str)
                    assert isinstance(result.effect_size, float)
                    if result.confidence_interval:
                        assert isinstance(result.confidence_interval, tuple)
                        assert len(result.confidence_interval) == 2
                        assert all(isinstance(x, float) for x in result.confidence_interval)

def test_equiboots_all_group_comparisons():
    """Test all-group comparisons through EquiBoots class."""
    # Create synthetic data with positive values
    n_samples = 100
    y_true = np.abs(np.random.normal(5, 1, n_samples))  # Ensure positive values
    y_pred = y_true + np.random.normal(0, 0.1, n_samples)
    y_prob = None
    
    # Create demographic data
    race = np.random.choice(
        ["white", "black", "asian", "hispanic"],
        n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    ).reshape(-1, 1)
    
    sex = np.random.choice(
        ["M", "F"],
        n_samples,
        p=[0.6, 0.4]
    ).reshape(-1, 1)
    
    fairness_df = pd.DataFrame(
        data=np.concatenate((race, sex), axis=1),
        columns=["race", "sex"]
    )
    
    # Initialize EquiBoots
    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
        reference_groups=["white", "M"],
        task="regression",
        bootstrap_flag=True,
        num_bootstraps=100,
        boot_sample_size=200,
        balanced=True,
        stratify_by_outcome=False
    )
    
    # Set fixed seeds for reproducibility
    eq.set_fix_seeds([42, 123, 222, 999])
    
    # Get metrics
    eq.grouper(groupings_vars=["race", "sex"])
    race_data = eq.slicer("race")
    race_metrics = eq.get_metrics(race_data)
    
    # Test with different configurations
    test_configs = [
        {
            "test_type": "mann_whitney",
            "alpha": 0.05,
            "adjust_method": "bonferroni"
        },
        {
            "test_type": "t_test",
            "alpha": 0.05,
            "adjust_method": "holm"
        },
        {
            "test_type": "ks_test",
            "alpha": 0.05,
            "adjust_method": "fdr_bh"
        },
        {
            "test_type": "permutation",
            "alpha": 0.05,
            "adjust_method": "none",
            "bootstrap_iterations": 1000
        }
    ]
    
    for config in test_configs:
        results = eq.analyze_statistical_significance(
            metric_dict=race_metrics,
            var_name="race",
            test_config=config
        )
        
        # Check that results are returned for all groups
        groups = ["black", "asian", "hispanic"]  # Exclude reference group
        assert set(results.keys()) == set(groups)
        
        # Check that each group has results for all metrics
        for group in groups:
            assert isinstance(results[group], dict)
            for metric, result in results[group].items():
                assert isinstance(result, StatTestResult)
                assert isinstance(result.statistic, float)
                assert isinstance(result.p_value, float)
                assert isinstance(result.is_significant, bool)
                assert isinstance(result.test_name, str)
                assert isinstance(result.effect_size, float)
                if result.confidence_interval:
                    assert isinstance(result.confidence_interval, tuple)
                    assert len(result.confidence_interval) == 2
                    assert all(isinstance(x, float) for x in result.confidence_interval) 