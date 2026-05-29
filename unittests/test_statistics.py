import copy
import warnings
import numpy as np
import pandas as pd
import pytest

from src.equiboots.StatisticalTester import StatTestResult, StatisticalTester


@pytest.fixture
def tester():
    return StatisticalTester()


@pytest.fixture
def chi_square_config():
    return {
        "test_type": "chi_square",
        "alpha": 0.05,
        "adjust_method": "none",
    }


@pytest.fixture
def bootstrap_config():
    return {
        "test_type": "bootstrap_test",
        "alpha": 0.05,
        "adjust_method": "none",
        "tail_type": "two_tailed",
        "confidence_level": 0.95,
        "metrics": ["Accuracy"],
    }


def test_stat_test_result_casts_numpy_bool_to_python_bool():
    result = StatTestResult(
        statistic=1.0,
        p_value=0.01,
        is_significant=np.bool_(True),
        test_name="example",
    )

    assert result.is_significant is True


@pytest.mark.parametrize(
    ("tail_type", "expected"),
    [
        ("two_tailed", (2.5, 97.5)),
        ("one_tail_less", (0, 5.0)),
        ("one_tail_greater", (95.0, 100)),
    ],
)
def test_get_ci_bounds_for_supported_tail_types(tester, tail_type, expected):
    config = {"tail_type": tail_type, "alpha": 0.05}

    assert tester.get_ci_bounds(config) == expected


def test_get_ci_bounds_rejects_unknown_tail_type(tester):
    with pytest.raises(ValueError, match="Must specify"):
        tester.get_ci_bounds({"tail_type": "sideways", "alpha": 0.05})


@pytest.mark.parametrize(
    ("data", "tail_type", "expected"),
    [
        ([-1, -1, 1, 1], "two_tailed", 0.5),
        ([1, 1, 1, -1], "two_tailed", 0.25),
        ([-1, -1, -1, 1], "one_tail_less", 0.75),
    ],
)
def test_calc_p_value_bootstrap(tester, data, tail_type, expected):
    config = {"tail_type": tail_type, "alpha": 0.05}

    assert tester.calc_p_value_bootstrap(data, config) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("metric", "expected_columns", "expected_values"),
    [
        ("Accuracy", ["TP + TN", "FP + FN"], [[11, 7], [17, 15]]),
        ("Precision", ["TP", "FP"], [[10, 5], [15, 11]]),
        ("Recall", ["TP", "FN"], [[10, 2], [15, 4]]),
        ("F1 Score", ["TP", "FP + FN"], [[10, 7], [15, 15]]),
        ("Specificity", ["TN", "FP"], [[1, 5], [2, 11]]),
        ("FN Rate", ["FN", "TP"], [[2, 10], [4, 15]]),
        ("FP Rate", ["FP", "TN"], [[5, 1], [11, 2]]),
        ("Predicted Prevalence", ["TP + FP", "FN + TN"], [[15, 3], [26, 6]]),
        ("Negative Predictive Value", ["TN", "FN"], [[1, 2], [2, 4]]),
    ],
)
def test_get_contingency_table_builds_expected_tables(
    tester, metric, expected_columns, expected_values
):
    data = pd.DataFrame(
        {
            "TP": [10, 15],
            "FP": [5, 11],
            "TN": [1, 2],
            "FN": [2, 4],
        },
        index=["reference", "group"],
    )

    table = tester.get_contingency_table(data.copy(), metric)

    assert list(table.columns) == expected_columns
    assert table.to_numpy().tolist() == expected_values


def test_chi_square_test_falls_back_to_fisher_for_small_2x2_tables(
    tester, chi_square_config
):
    metrics = {
        "reference": {"TP": 1, "FP": 1, "TN": 1, "FN": 1},
        "group": {"TP": 1, "FP": 1, "TN": 1, "FN": 1},
    }

    results = tester._chi_square_test(metrics, chi_square_config)

    assert set(results) == set(tester.METRIC_LIST)
    assert all(result.test_name == "Fisher's Exact Test" for result in results.values())
    assert all(np.isnan(result.statistic) for result in results.values())
    assert all(result.p_value == pytest.approx(1.0) for result in results.values())
    assert all(result.is_significant is False for result in results.values())


def test_chi_square_fishers_exact_fallback_2x2():
    """2x2 table with >20% expected cells <5 should fall back to Fisher's exact."""
    tester = StatisticalTester()
    metrics = {
        "ref": {"TP": 2, "FP": 1, "TN": 5, "FN": 1},
        "groupA": {"TP": 1, "FP": 0, "TN": 4, "FN": 1},
    }
    config = {"alpha": 0.05}

    result = tester._chi_square_test(metrics, config)

    for metric_name, test_result in result.items():
        assert (
            test_result.test_name == "Fisher's Exact Test"
        ), f"{metric_name}: expected Fisher's Exact Test, got {test_result.test_name}"
        assert np.isnan(
            test_result.statistic
        ), f"{metric_name}: chi2 statistic should be NaN when Fisher's is used"
        assert 0.0 <= test_result.p_value <= 1.0


def test_chi_square_warns_on_low_expected_Kx2():
    """K x 2 table (K>2) with >20% expected cells <5 should emit a Cochran warning."""
    tester = StatisticalTester()
    metrics = {
        "ref": {"TP": 2, "FP": 1, "TN": 5, "FN": 1},
        "groupA": {"TP": 1, "FP": 0, "TN": 4, "FN": 1},
        "groupB": {"TP": 1, "FP": 1, "TN": 3, "FN": 0},
    }
    config = {"alpha": 0.05}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tester._chi_square_test(metrics, config)

    cochran_warnings = [w for w in caught if "Cochran" in str(w.message)]
    assert len(cochran_warnings) > 0, "Expected Cochran's rule warning, got none"


def test_chi_square_normal_path_large_counts():
    """Healthy expected counts should use plain chi-square, no Fisher, no warning."""
    tester = StatisticalTester()
    metrics = {
        "ref": {"TP": 100, "FP": 50, "TN": 200, "FN": 100},
        "groupA": {"TP": 80, "FP": 40, "TN": 180, "FN": 90},
    }
    config = {"alpha": 0.05}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = tester._chi_square_test(metrics, config)

    cochran_warnings = [w for w in caught if "Cochran" in str(w.message)]
    assert (
        len(cochran_warnings) == 0
    ), "Got unexpected Cochran warning for healthy counts"

    for metric_name, test_result in result.items():
        assert (
            test_result.test_name == "Chi-Square Test"
        ), f"{metric_name}: expected Chi-Square Test, got {test_result.test_name}"
        assert not np.isnan(
            test_result.statistic
        ), f"{metric_name}: chi2 statistic should be a real number"


def test_analyze_metrics_runs_omnibus_then_pairwise_and_effect_sizes(
    tester, chi_square_config
):
    metrics = {
        "reference": {"TP": 90, "FP": 10, "TN": 90, "FN": 10},
        "different": {"TP": 10, "FP": 90, "TN": 10, "FN": 90},
        "similar": {"TP": 88, "FP": 12, "TN": 91, "FN": 9},
    }

    results = tester.analyze_metrics(
        metrics,
        reference_group="reference",
        test_config=chi_square_config,
        task="binary_classification",
    )

    assert set(results) == {"omnibus", "different", "similar"}
    assert results["omnibus"]["Recall"].is_significant is True
    assert results["omnibus"]["Recall"].effect_size > 0
    assert results["different"]["Recall"].is_significant is True
    assert results["different"]["Recall"].effect_size > 0
    assert results["similar"]["Recall"].is_significant is False
    assert results["similar"]["Recall"].effect_size is None
    assert results["omnibus"]["Predicted Prevalence"].is_significant is False
    assert results["omnibus"]["Predicted Prevalence"].effect_size is None


def test_analyze_metrics_skips_pairwise_when_omnibus_has_no_significance(
    tester, chi_square_config
):
    metrics = {
        "reference": {"TP": 50, "FP": 10, "TN": 40, "FN": 8},
        "group": {"TP": 50, "FP": 10, "TN": 40, "FN": 8},
        "other": {"TP": 50, "FP": 10, "TN": 40, "FN": 8},
    }

    results = tester.analyze_metrics(
        metrics,
        reference_group="reference",
        test_config=chi_square_config,
        task="binary_classification",
    )

    assert set(results) == {"omnibus"}
    assert all(result.is_significant is False for result in results["omnibus"].values())


def test_analyze_bootstrapped_metrics_aggregates_differences(tester, bootstrap_config):
    group_a_values = np.linspace(0.1, 0.2, 5000)
    group_b_values = np.linspace(-0.2, -0.1, 5000)
    differences = [
        {
            "reference": {"Accuracy": 0.0},
            "group_a": {"Accuracy": group_a_value},
            "group_b": {"Accuracy": group_b_value},
        }
        for group_a_value, group_b_value in zip(group_a_values, group_b_values)
    ]

    results = tester.analyze_metrics(
        metrics_data=[],
        reference_group="reference",
        test_config=bootstrap_config,
        differences=copy.deepcopy(differences),
    )

    assert set(results) == {"group_a", "group_b"}
    assert results["group_a"]["Accuracy"].statistic == pytest.approx(0.15)
    assert results["group_b"]["Accuracy"].statistic == pytest.approx(-0.15)
    assert results["group_a"]["Accuracy"].confidence_interval == pytest.approx(
        (np.percentile(group_a_values, 2.5), np.percentile(group_a_values, 97.5))
    )
    assert results["group_a"]["Accuracy"].is_significant is True
    assert results["group_b"]["Accuracy"].is_significant is True


def test_adjust_p_values_for_bootstrap_results_adjusts_across_all_group_metrics(tester):
    results = {
        "group_a": {
            "Accuracy": StatTestResult(0.1, 0.01, True, "bootstrap_mean"),
            "Recall": StatTestResult(0.1, 0.04, True, "bootstrap_mean"),
        },
        "group_b": {
            "Accuracy": StatTestResult(0.1, 0.03, True, "bootstrap_mean"),
        },
    }

    adjusted = tester._adjust_p_values(
        results,
        method="bonferroni",
        alpha=0.05,
        boot=True,
    )

    assert adjusted["group_a"]["Accuracy"].p_value == pytest.approx(0.03)
    assert adjusted["group_a"]["Accuracy"].is_significant == True
    assert adjusted["group_b"]["Accuracy"].p_value == pytest.approx(0.09)
    assert adjusted["group_b"]["Accuracy"].is_significant == False
    assert adjusted["group_a"]["Recall"].p_value == pytest.approx(0.12)
    assert adjusted["group_a"]["Recall"].is_significant == False


def test_adjust_p_values_for_chi_square_results_excludes_omnibus(tester):
    results = {
        "omnibus": {
            "Recall": StatTestResult(1.0, 0.001, True, "Chi-Square Test"),
        },
        "group_a": {
            "Recall": StatTestResult(1.0, 0.01, True, "Chi-Square Test"),
        },
        "group_b": {
            "Recall": StatTestResult(1.0, 0.04, True, "Chi-Square Test"),
        },
    }

    adjusted = tester._adjust_p_values(
        results,
        method="bonferroni",
        alpha=0.05,
        boot=False,
    )

    assert adjusted["omnibus"]["Recall"].p_value == pytest.approx(0.001)
    assert adjusted["group_a"]["Recall"].p_value == pytest.approx(0.02)
    assert adjusted["group_a"]["Recall"].is_significant == True
    assert adjusted["group_b"]["Recall"].p_value == pytest.approx(0.08)
    assert adjusted["group_b"]["Recall"].is_significant == False


def test_validate_config_rejects_invalid_test_type(tester):
    config = {
        "test_type": "not_a_test",
        "alpha": 0.05,
        "adjust_method": "none",
    }

    with pytest.raises(ValueError, match="Invalid test type"):
        tester._validate_config(config)


def test_validate_config_rejects_invalid_adjustment_method(tester):
    config = {
        "test_type": "chi_square",
        "alpha": 0.05,
        "adjust_method": "not_a_method",
    }

    with pytest.raises(ValueError, match="Invalid adjustment method"):
        tester._validate_config(config)


def test_analyze_metrics_rejects_non_bootstrapped_unsupported_task(
    tester, chi_square_config
):
    with pytest.raises(ValueError, match="Task not supported"):
        tester.analyze_metrics(
            metrics_data={"reference": {"TP": 1, "FP": 1, "TN": 1, "FN": 1}},
            reference_group="reference",
            test_config=chi_square_config,
            task="regression",
        )
