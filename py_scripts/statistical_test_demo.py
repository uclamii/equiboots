import numpy as np
import pandas as pd
from equiboots import EquiBoots
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def create_test_configs() -> Dict[str, Dict[str, Any]]:
    """Create different test configurations to demonstrate flexibility."""
    return {
        "default": None,  # Use default settings
        
        "liberal": {
            "test_type": "permutation",  # More sensitive test
            "alpha": 0.05,
            "adjust_method": "none",  # No adjustment
            "bootstrap_iterations": 1000,
            "confidence_level": 0.95,
            "alternative": "two-sided"
        },
        
        "non_liberal": {
            "test_type": "mann_whitney",  # More conservative test
            "alpha": 0.01,  # More conservative alpha
            "adjust_method": "bonferroni",  # Most conservative adjustment
            "alternative": "two-sided"
        }
    }


def test_statistical_significance(task: str = "binary_classification", n_samples: int = 1000):
    """
    Demonstrate the statistical significance testing functionality of EquiBoots.
    
    Parameters:
    -----------
    task : str
        Type of ML task to test
    n_samples : int
        Number of samples to generate for testing
    """
    # Generate synthetic data based on task
    if task == "binary_classification":
        # Create imbalanced data to demonstrate significant differences
        y_prob = np.zeros(n_samples)
        y_prob[:n_samples//2] = np.random.RandomState(3).rand(n_samples//2) * 0.4 + 0.1
        y_prob[n_samples//2:] = np.random.RandomState(4).rand(n_samples//2) * 0.4 + 0.6
        y_pred = (y_prob > 0.5).astype(int)
        y_true = np.zeros(n_samples)
        y_true[:n_samples//2] = np.random.RandomState(30).randint(0, 2, n_samples//2)
        y_true[n_samples//2:] = np.random.RandomState(31).randint(0, 2, n_samples//2)
        
    elif task == "multi_class_classification":
        n_classes = 3
        # Create class imbalance
        y_prob = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            start = (i * n_samples) // n_classes
            end = ((i + 1) * n_samples) // n_classes
            y_prob[start:end, i] = np.random.RandomState(i).rand(end - start) * 0.5 + 0.5
        y_prob /= y_prob.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.random.RandomState(30).randint(0, n_classes, n_samples)
        
    elif task == "regression":
        # Generate synthetic regression data with positive values
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 5)
        true_coef = np.array([1.5, -2.0, 0.5, 1.0, -0.5])
        # Add a large constant to ensure positive values
        y_true = np.abs(X @ true_coef) + np.random.RandomState(42).normal(5, 1, n_samples)
        y_pred = y_true + np.random.RandomState(6).normal(0, 0.5, n_samples)
        y_prob = None
        
    elif task == "multi_label_classification":
        n_classes = 3
        y_true = np.zeros((n_samples, n_classes))
        # Create imbalanced multi-label data
        y_true[:n_samples//3, 0] = 1
        y_true[n_samples//3:2*n_samples//3, 1] = 1
        y_true[2*n_samples//3:, 2] = 1
        y_prob = np.random.RandomState(3).rand(n_samples, n_classes)
        y_prob /= y_prob.sum(axis=1, keepdims=True)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        raise ValueError("Invalid task")

    # Create demographic data with intentional disparities
    np.random.seed(42)
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
        task=task,
        bootstrap_flag=True,
        num_bootstraps=100,
        boot_sample_size=200,
        balanced=True,
        stratify_by_outcome=True if task != "regression" else False
    )

    # Set fixed seeds for reproducibility
    eq.set_fix_seeds([42, 123, 222, 999])

    # List available tests and methods
    print("\nAvailable Statistical Tests:")
    for test, desc in eq.list_available_tests().items():
        print(f"- {test}: {desc}")
    
    print("\nAvailable Adjustment Methods:")
    for method, desc in eq.list_adjustment_methods().items():
        print(f"- {method}: {desc}")

    # Get metrics
    eq.grouper(groupings_vars=["race", "sex"])
    race_data = eq.slicer("race")
    race_metrics = eq.get_metrics(race_data)
    
    # Test different statistical configurations
    test_configs = create_test_configs()
    
    for config_name, config in test_configs.items():
        print(f"\n{'='*20} Testing {config_name} configuration {'='*20}")
        
        # Perform statistical testing
        results = eq.analyze_statistical_significance(
            metric_dict=race_metrics,
            var_name="race",
            test_config=config
        )
        
        # Print results
        print(f"\nResults for {task} with {config_name} configuration:")
        for group, group_results in results.items():
            if group != "white":  # Skip reference group
                print(f"\n{group} vs white:")
                for metric, test_result in group_results.items():
                    print(f"\n{metric}:")
                    print(f"  Test: {test_result.test_name}")
                    print(f"  Statistic: {test_result.statistic:.3f}")
                    print(f"  P-value: {test_result.p_value:.3f}")
                    print(f"  Significant: {test_result.is_significant}")
                    print(f"  Effect size: {test_result.effect_size:.3f}")
                    if test_result.confidence_interval:
                        ci_lower, ci_upper = test_result.confidence_interval
                        print(f"  95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
        
        # Visualize results
        plot_metrics_with_significance(
            race_metrics[0],
            results,
            f"Race-based Metrics ({config_name} configuration)"
        )


def plot_metrics_with_significance(
    metrics: dict,
    stat_results: dict,
    title: str
):
    """Create an enhanced bar plot of metrics with significance indicators."""
    # Prepare data for plotting
    plot_data = []
    for group, group_metrics in metrics.items():
        for metric, value in group_metrics.items():
            if isinstance(value, (int, float)):
                significance = ""
                effect_size = None
                p_value = None
                ci_lower = None
                ci_upper = None
                if group in stat_results and metric in stat_results[group]:
                    result = stat_results[group][metric]
                    if result.is_significant:
                        significance = "*"
                    effect_size = result.effect_size
                    p_value = result.p_value
                    if result.confidence_interval:
                        ci_lower, ci_upper = result.confidence_interval
                plot_data.append({
                    "Group": group,
                    "Metric": metric,
                    "Value": value,
                    "Significance": significance,
                    "Effect Size": effect_size,
                    "P-value": p_value,
                    "CI Lower": ci_lower,
                    "CI Upper": ci_upper
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    # Main plot
    g = sns.barplot(data=df, x="Metric", y="Value", hue="Group")
    
    # Add significance markers and effect sizes
    for i, row in df.iterrows():
        if row["Significance"]:
            x_pos = df[df["Metric"] == row["Metric"]].index[0] % len(df["Metric"].unique())
            y_pos = row["Value"]
            
            # Add star for significance
            g.text(
                x_pos,
                y_pos,
                row["Significance"],
                ha='center',
                va='bottom',
                color='red',
                fontweight='bold'
            )
            
            # Add effect size and p-value
            if row["Effect Size"] is not None:
                effect_size_text = f'd={row["Effect Size"]:.2f}\np={row["P-value"]:.3f}'
                if row["CI Lower"] is not None and row["CI Upper"] is not None:
                    effect_size_text += f'\nCI: ({row["CI Lower"]:.2f}, {row["CI Upper"]:.2f})'
                g.text(
                    x_pos,
                    y_pos,
                    effect_size_text,
                    ha='center',
                    va='top',
                    fontsize=8
                )
    
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def test_all_group_comparisons(task: str = "binary_classification", n_samples: int = 1000):
    """
    Demonstrate the all-group comparisons functionality of EquiBoots.
    
    Parameters:
    -----------
    task : str
        Type of ML task to test
    n_samples : int
        Number of samples to generate for testing
    """
    # Generate synthetic data based on task
    if task == "binary_classification":
        # Create imbalanced data to demonstrate significant differences
        y_prob = np.zeros(n_samples)
        y_prob[:n_samples//2] = np.random.RandomState(3).rand(n_samples//2) * 0.4 + 0.1
        y_prob[n_samples//2:] = np.random.RandomState(4).rand(n_samples//2) * 0.4 + 0.6
        y_pred = (y_prob > 0.5).astype(int)
        y_true = np.zeros(n_samples)
        y_true[:n_samples//2] = np.random.RandomState(30).randint(0, 2, n_samples//2)
        y_true[n_samples//2:] = np.random.RandomState(31).randint(0, 2, n_samples//2)
        
    elif task == "multi_class_classification":
        n_classes = 3
        # Create class imbalance
        y_prob = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            start = (i * n_samples) // n_classes
            end = ((i + 1) * n_samples) // n_classes
            y_prob[start:end, i] = np.random.RandomState(i).rand(end - start) * 0.5 + 0.5
        y_prob /= y_prob.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.random.RandomState(30).randint(0, n_classes, n_samples)
        
    elif task == "regression":
        # Generate synthetic regression data with positive values
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 5)
        true_coef = np.array([1.5, -2.0, 0.5, 1.0, -0.5])
        # Add a large constant to ensure positive values
        y_true = np.abs(X @ true_coef) + np.random.RandomState(42).normal(5, 1, n_samples)
        y_pred = y_true + np.random.RandomState(6).normal(0, 0.5, n_samples)
        y_prob = None
        
    elif task == "multi_label_classification":
        n_classes = 3
        y_true = np.zeros((n_samples, n_classes))
        # Create imbalanced multi-label data
        y_true[:n_samples//3, 0] = 1
        y_true[n_samples//3:2*n_samples//3, 1] = 1
        y_true[2*n_samples//3:, 2] = 1
        y_prob = np.random.RandomState(3).rand(n_samples, n_classes)
        y_prob /= y_prob.sum(axis=1, keepdims=True)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        raise ValueError("Invalid task")

    # Create demographic data with intentional disparities
    np.random.seed(42)
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
        task=task,
        bootstrap_flag=True,
        num_bootstraps=100,
        boot_sample_size=200,
        balanced=True,
        stratify_by_outcome=True if task != "regression" else False
    )

    # Set fixed seeds for reproducibility
    eq.set_fix_seeds([42, 123, 222, 999])

    # Get metrics
    eq.grouper(groupings_vars=["race", "sex"])
    race_data = eq.slicer("race")
    race_metrics = eq.get_metrics(race_data)
    
    # Test different statistical configurations for all-group comparisons
    test_configs = {
        "default": None,  # Use default settings
        
        "liberal": {
            "test_type": "permutation",  # More sensitive test
            "alpha": 0.05,
            "adjust_method": "none",  # No adjustment
            "bootstrap_iterations": 1000,
            "confidence_level": 0.95,
            "alternative": "two-sided"
        },
        
        "non_liberal": {
            "test_type": "mann_whitney",  # More conservative test
            "alpha": 0.01,  # More conservative alpha
            "adjust_method": "bonferroni",  # Most conservative adjustment
            "alternative": "two-sided"
        }
    }
    
    for config_name, config in test_configs.items():
        print(f"\n{'='*20} Testing {config_name} configuration for all-group comparisons {'='*20}")
        
        # Perform statistical testing
        results = eq.analyze_statistical_significance(
            metric_dict=race_metrics,
            var_name="race",
            test_config=config
        )
        
        # Print results
        print(f"\nResults for {task} with {config_name} configuration:")
        for group, group_results in results.items():
            if group != "white":  # Skip reference group
                print(f"\n{group} vs white:")
                for metric, test_result in group_results.items():
                    print(f"\n{metric}:")
                    print(f"  Test: {test_result.test_name}")
                    print(f"  Statistic: {test_result.statistic:.3f}")
                    print(f"  P-value: {test_result.p_value:.3f}")
                    print(f"  Significant: {test_result.is_significant}")
                    print(f"  Effect size: {test_result.effect_size:.3f}")
                    if test_result.confidence_interval:
                        ci_lower, ci_upper = test_result.confidence_interval
                        print(f"  95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
        
        # Visualize results
        plot_metrics_with_significance(
            race_metrics[0],
            results,
            f"All Group Comparisons ({config_name} configuration)"
        )


def plot_all_group_comparisons(
    metrics: dict,
    stat_results: dict,
    title: str
):
    """Create a heatmap visualization of all group comparisons."""
    # Prepare data for plotting
    groups = list(metrics.keys())
    metrics_list = [m for m in metrics[groups[0]].keys() if isinstance(metrics[groups[0]][m], (int, float))]
    
    # Create a figure with subplots for each metric
    n_metrics = len(metrics_list)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5*n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics_list):
        # Create a matrix for the heatmap
        heatmap_data = np.zeros((len(groups), len(groups)))
        significance_mask = np.zeros((len(groups), len(groups)), dtype=bool)
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if group1 != group2 and group1 in stat_results and group2 in stat_results[group1]:
                    result = stat_results[group1][group2][metric]
                    heatmap_data[i, j] = result.effect_size
                    significance_mask[i, j] = result.is_significant
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            mask=np.eye(len(groups), dtype=bool),  # Mask diagonal
            ax=ax
        )
        
        # Add significance markers
        for i in range(len(groups)):
            for j in range(len(groups)):
                if significance_mask[i, j]:
                    ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center", color="black")
        
        ax.set_title(f"{metric} - Effect Sizes")
        ax.set_xticklabels(groups, rotation=45)
        ax.set_yticklabels(groups, rotation=0)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test with different tasks
    tasks = [
        "binary_classification",
        "multi_class_classification",
        "regression",
        "multi_label_classification"
    ]
    
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Testing {task}")
        print(f"{'='*50}")
        test_statistical_significance(task)
        test_all_group_comparisons(task)  # Add all-group comparisons test 