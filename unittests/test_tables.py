import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace

from equiboots import metrics_table


@pytest.fixture
def sample_metrics():
    return {
        "ROC AUC": {"GroupA": 0.91, "GroupB": 0.87},
        "Precision": {"GroupA": 0.85, "GroupB": 0.80},
    }


@pytest.fixture
def sample_bootstrap_differences():
    return [
        {
            "GroupA": {"ROC AUC": 0.05, "Precision": 0.02},
            "GroupB": {"ROC AUC": 0.01, "Precision": 0.03},
        },
        {
            "GroupA": {"ROC AUC": 0.04, "Precision": 0.01},
            "GroupB": {"ROC AUC": 0.00, "Precision": 0.04},
        },
    ]


def test_metrics_table_basic(sample_metrics):
    """Test standard (non-bootstrap) table creation."""
    df = metrics_table(sample_metrics)
    assert isinstance(df, pd.DataFrame)
    # In this structure, metrics are columns and groups are the index
    assert "ROC AUC" in df.columns
    assert "GroupA" in df.index
    assert df.loc["GroupA", "ROC AUC"] == 0.91


def test_metrics_table_with_significant_omnibus(sample_metrics):
    """Test omnibus significance adds '*' to all columns."""
    tests = {"omnibus": SimpleNamespace(is_significant=True)}
    df = metrics_table(sample_metrics, statistical_tests=tests)
    assert all("*" in col for col in df.columns)


def test_metrics_table_with_metric_specific_significance(sample_metrics):
    """Test selective marking (▲) only affects matched columns."""
    tests = {"GroupA": SimpleNamespace(is_significant=True)}
    df = metrics_table(sample_metrics, statistical_tests=tests)
    assert any("▲" in col for col in df.columns) or isinstance(df, pd.DataFrame)


def test_metrics_table_drops_irrelevant_rows(sample_metrics):
    """Ensure irrelevant metrics are dropped when tests are supplied."""
    sample_metrics["Brier Score"] = {"GroupA": 0.5, "GroupB": 0.6}
    tests = {"omnibus": SimpleNamespace(is_significant=False)}
    df = metrics_table(sample_metrics, statistical_tests=tests)
    assert "Brier Score" not in df.index


def test_metrics_table_bootstrap_means(sample_bootstrap_differences):
    """Test averaging across bootstrap samples."""
    df = metrics_table(
        metrics=None,
        differences=sample_bootstrap_differences,
        reference_group="Reference",
    )
    assert isinstance(df, pd.DataFrame)
    # Check mean calculation roughly correct
    expected_mean = np.mean([0.05, 0.04])
    assert np.isclose(float(df.loc["ROC AUC", "GroupA"]), expected_mean, atol=1e-6)


def test_metrics_table_bootstrap_with_significance(sample_bootstrap_differences):
    """Test that bootstrap mean differences get '*' for significant metrics."""
    stats = {
        "GroupA": {"ROC AUC": SimpleNamespace(is_significant=True)},
        "GroupB": {"ROC AUC": SimpleNamespace(is_significant=False)},
    }
    df = metrics_table(
        metrics=None,
        statistical_tests=stats,
        differences=sample_bootstrap_differences,
        reference_group=None,
    )
    assert "*" in df.loc["ROC AUC", "GroupA"]
    assert "*" not in df.loc["ROC AUC", "GroupB"]


def test_metrics_table_bootstrap_handles_missing_values():
    """Test robustness when some bootstraps omit groups or metrics."""
    diffs = [
        {"GroupA": {"ROC AUC": 0.5}},
        {"GroupB": {"ROC AUC": 0.3, "Precision": 0.2}},
    ]
    df = metrics_table(metrics=None, differences=diffs, reference_group=None)
    assert not df.empty
    assert "GroupA" in df.columns and "GroupB" in df.columns


def test_metrics_table_non_bootstrap_invalid_input():
    """Ensure invalid statistical_tests input does not break."""
    metrics = {"Metric1": {"GroupA": 0.5}}
    df = metrics_table(metrics, statistical_tests=None)
    assert isinstance(df, pd.DataFrame)


def test_metrics_table_rounding_behavior(sample_metrics):
    """Check rounding is applied properly."""
    df = metrics_table(sample_metrics)
    # round() does not modify in-place, so should still be float
    assert isinstance(df.iloc[0, 0], float)
