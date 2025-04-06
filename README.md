[![Downloads](https://pepy.tech/badge/equiboots)](https://pepy.tech/project/equiboots) [![PyPI](https://img.shields.io/pypi/v/equiboots.svg)](https://pypi.org/project/equiboots/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15086941.svg)](https://doi.org/10.5281/zenodo.15086941)


The `equiboots` library is a fairness-aware model evaluation toolkit designed to audit performance disparities across demographic groups. It provides robust, bootstrapped metrics for binary, multi-class, and multi-label classification, as well as regression models. The library supports group-wise performance slicing, fairness diagnostics, and customizable visualizations to support equitable AI/ML development.

`equiboots` is particularly useful in clinical, social, and policy domains where transparency, bias mitigation, and outcome fairness are critical for responsible deployment.

## Prerequisites

Before installing `equiboots`, ensure your system meets the following requirements:

## Python Version

`equiboots` requires **Python 3.7.4 or higher**. Specific dependency versions vary depending on your Python version.

## Dependencies

The following dependencies will be automatically installed with `equiboots`:

`fastparquet==2024.11.0`  
`matplotlib==3.10.1`  
`model_tuner==0.0.29b1`  
`numpy==1.23.5`  
`pandas==1.5.0`  
`pytest==8.3.5`  
`pytest_cov==6.0.0`  
`scikit-learn==1.5.1`  
`scipy==1.14.0`  
`seaborn==0.13.2`  
`tqdm==4.66.4`  
`ucimlrepo==0.0.7`  
`statsmodels>=0.13.0`  

## 💾 Installation

You can install `equiboots` directly from PyPI:

```bash
pip install equiboots
```

## 📄 Official Documentation

https://uclamii.github.io/equiboots


## 🌐 Author Website

https://www.mii.ucla.edu/

## ⚖️ License

`equiboots` is distributed under the Apache License. See [LICENSE](https://github.com/uclamii/equiboots?tab=Apache-2.0-1-ov-file) for more information.

## 📚 Citing `equiboots`

If you use `equiboots` in your research or projects, please consider citing it.

```bibtex
@software{shpaner_2025_15086941,
  author       = {Shpaner, Leonid and
                  Funnell, Arthur and
                  Rahrooh, Al and
                  Petousis, Panayiotis},
  title        = {EquiBoots},
  month        = mar,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.0.0a3},
  doi          = {10.5281/zenodo.15086941},
  url          = {https://doi.org/10.5281/zenodo.15086941}
}
```

## Support

If you have any questions or issues with `equiboots`, please open an issue on this [GitHub repository](https://github.com/uclamii/equiboots/).

## Acknowledgements

This work was supported by the UCLA Medical Informatics Institute (MII) and the Clinical and Translational Science Institute (CTSI). Special thanks to Dr. Alex Bui for his invaluable guidance and support, and to Panayiotis Petousis, PhD, for his contributions to this codebase.

# StatisticalTester

The `StatisticalTester` class provides a comprehensive suite of statistical tests for analyzing differences between groups in machine learning metrics. It supports various statistical tests and multiple comparison adjustments.

## Features

- Multiple statistical test implementations
- Support for different types of data (continuous, discrete)
- Multiple comparison adjustments
- Effect size calculations
- Confidence interval estimation
- Flexible configuration options
- Support for both reference group and all-group comparisons

## Available Statistical Tests

1. **Mann-Whitney U Test**
   - Non-parametric test for comparing two independent samples
   - Suitable for ordinal or continuous data
   - Robust to non-normal distributions

2. **Welch's t-test**
   - Parametric test for comparing two independent samples
   - Handles unequal variances
   - Assumes approximately normal distributions

3. **Kolmogorov-Smirnov Test**
   - Non-parametric test for comparing distributions
   - Tests for differences in shape and location
   - Suitable for continuous data

4. **Bootstrap Test**
   - Resampling-based test for any test statistic
   - Provides confidence intervals
   - Makes minimal assumptions about data distribution

5. **Wilcoxon Signed-Rank Test**
   - Non-parametric test for paired samples
   - Suitable for ordinal or continuous data
   - Handles non-normal distributions

6. **Permutation Test**
   - Customizable test for any test statistic
   - Makes minimal assumptions
   - Flexible for various comparison scenarios

## Multiple Comparison Adjustments

1. **Bonferroni Correction**
   - Conservative adjustment method
   - Controls family-wise error rate
   - Suitable for small number of comparisons

2. **Benjamini-Hochberg FDR**
   - Less conservative than Bonferroni
   - Controls false discovery rate
   - Suitable for large number of comparisons

3. **Holm-Bonferroni Method**
   - Step-down procedure
   - More powerful than Bonferroni
   - Controls family-wise error rate

## Usage Examples

### Reference Group Comparison
```python
from equiboots.StatisticalTester import StatisticalTester

# Initialize the tester
tester = StatisticalTester()

# Example data
metrics_data = {
    'group1': {'metric1': 0.8, 'metric2': 0.6},
    'group2': {'metric1': 0.9, 'metric2': 0.7},
    'group3': {'metric1': 0.85, 'metric2': 0.65}
}

# Perform tests against reference group
results = tester.analyze_metrics(
    metrics_data,
    reference_group='group1',
    test_config={"alpha": 0.05}
)
```

### All Group Comparisons
```python
# Perform tests between all groups
all_results = tester.analyze_all_group_comparisons(
    metrics_data,
    test_config={
        "test_type": "mann_whitney",
        "alpha": 0.05,
        "adjust_method": "bonferroni"  # Important for multiple comparisons
    }
)

# Access results for specific group pair
group1_vs_group2 = all_results['group1']['group2']
print(f"Metric1 comparison: {group1_vs_group2['metric1'].p_value}")
```

## Configuration Options

The statistical tests accept a configuration dictionary with the following options:

```python
config = {
    "alpha": 0.05,                    # Significance level
    "alternative": "two-sided",       # Alternative hypothesis
    "bootstrap_iterations": 1000,     # Number of bootstrap samples
    "confidence_level": 0.95,         # Confidence level for intervals
    "adjust_method": "none",          # Multiple comparison adjustment
    "test_statistic": None           # Custom test statistic function
}
```

## Effect Size Measures

- Cohen's d for parametric tests
- Rank-biserial correlation for non-parametric tests
- Custom effect size calculations for specialized tests

## Confidence Intervals

- Bootstrap-based confidence intervals
- Parametric confidence intervals where appropriate
- Custom interval calculations for specialized tests

## Best Practices

1. **Test Selection**
   - Use parametric tests for normally distributed data
   - Use non-parametric tests for skewed or ordinal data
   - Consider sample size when choosing tests

2. **Multiple Comparisons**
   - Use appropriate adjustment methods based on the number of comparisons
   - Consider the trade-off between power and type I error control
   - Document the adjustment method used
   - When comparing all groups, use more conservative adjustments (e.g., Bonferroni)

3. **Effect Size Interpretation**
   - Report effect sizes along with p-values
   - Use standardized effect sizes for comparability
   - Consider practical significance alongside statistical significance

4. **Assumptions**
   - Check test assumptions before application
   - Use robust tests when assumptions are violated
   - Document any deviations from standard assumptions

5. **All-Group Comparisons**
   - Use when you need to understand relationships between all groups
   - Be aware of increased multiple comparison burden
   - Consider using more conservative alpha levels
   - Document all significant findings, not just those against reference
