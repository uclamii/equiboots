import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp
import os
from sklearn.utils import resample
from aequitas.audit import Audit
import matplotlib.ticker as ticker


################################################################################
########################## Bootstrapped Bias & Fairness ########################
################################################################################


def perform_bootstrapped_audit(
    df,
    seeds,
    n_iterations,
    sample_size,
    stratify_columns,
    categorical_columns,
    score_column,
    label_column,
    bootstrap_method="stratified",
    calculate_statistics=False,
    attribute_name=None,
    attribute_value=None,
    return_disparity_metrics=False,
):
    """
    Perform bootstrapped audit on the given dataframe using the specified
    bootstrap method.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    seeds (list): List of random seeds for reproducibility.
    n_iterations (int): Number of bootstrapping iterations.
    sample_size (int): Sample size for each iteration.
    stratify_columns (list): Columns to stratify by.
    categorical_columns (list): Columns to be used in the audit.
    score_column (str): The column name of the score.
    label_column (str): The column name of the label.
    bootstrap_method (str): The bootstrap method to use ('stratified' or 'balanced').
    calculate_statistics (bool): Whether to calculate disparity statistics.
    attribute_name (str): The attribute name for disparity statistics calculation.
    attribute_value (str): The attribute value for disparity statistics calculation.
    return_disparity_metrics (bool): Whether to return the disparity metrics DataFrame.

    Returns:
    pd.DataFrame: Combined metrics results from the audits.
    dict (optional): Disparity statistics if calculate_statistics is True.
    """

    # Initialize the results dictionary
    results_dict = {}

    def stratified_bootstrap(
        df,
        seeds,
        n_iterations,
        sample_size,
        stratify_columns,
    ):
        """
        Perform stratified bootstrap sampling on the dataframe.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        seeds (list): List of random seeds for reproducibility.
        n_iterations (int): Number of bootstrapping iterations.
        sample_size (int): Sample size for each iteration.
        stratify_columns (list): Columns to stratify by.

        Returns:
        list: A list of bootstrapped samples.
        """

        bootstrapped_samples = []
        strata = df.groupby(stratify_columns)

        for i in tqdm(range(n_iterations), desc="Bootstrapping iterations"):
            stratified_sample = []

            for _, group in strata:
                if len(group) == 0:
                    continue

                n_samples = max(1, int(len(group) * sample_size / len(df)))

                sampled_group = resample(
                    group,
                    replace=True,
                    n_samples=n_samples,
                    random_state=seeds[i % len(seeds)],
                )

                stratified_sample.append(sampled_group)

            if stratified_sample:
                bootstrapped_samples.append(pd.concat(stratified_sample))

        return bootstrapped_samples

    def balanced_bootstrap(
        df,
        seeds,
        n_iterations,
        sample_size,
        group_columns,
    ):
        """
        Perform balanced bootstrap sampling on the dataframe.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        seeds (list): List of random seeds for reproducibility.
        n_iterations (int): Number of bootstrapping iterations.
        sample_size (int): Sample size for each iteration.
        group_columns (list): Columns to group by.

        Returns:
        list: A list of bootstrapped samples.
        """

        bootstrapped_samples = []
        groups = df.groupby(group_columns)
        n_groups = df[group_columns].nunique()

        for indx in tqdm(range(n_iterations), desc="Bootstrapping iterations"):
            samples = []

            for _, group in groups:
                if len(group) == 0:
                    continue

                n_samples = max(1, int(sample_size / n_groups))

                sampled_group = resample(
                    group,
                    replace=True,
                    n_samples=n_samples,
                    random_state=seeds[indx],
                )

                samples.append(sampled_group)

            if samples:
                bootstrapped_samples.append(pd.concat(samples))

        return bootstrapped_samples

    def calculate_disparity_statistics(
        all_metrics,
        attribute_name,
        attribute_value,
    ):
        """
        Calculate disparity statistics for the given attribute and value.

        Parameters:
        all_metrics (pd.DataFrame): Dataframe containing all metrics results
        from the audits.
        attribute_name (str): The attribute name for disparity statistics
        calculation.
        attribute_value (str): The attribute value for disparity statistics
        calculation.

        Returns:
        dict: A dictionary containing disparity statistics.
        """

        metrics_columns = [
            "accuracy",
            "tpr",
            "tnr",
            "for",
            "fdr",
            "fpr",
            "fnr",
            "npv",
            "precision",
            "ppr",
            "pprev",
            "prev",
            "ppr_disparity",
            "pprev_disparity",
            "precision_disparity",
            "fdr_disparity",
            "for_disparity",
            "fpr_disparity",
            "fnr_disparity",
            "tpr_disparity",
            "tnr_disparity",
            "npv_disparity",
        ]
        disparity_stats = {}

        for metric in metrics_columns:
            if metric not in all_metrics.columns:
                print(f"Skipping metric '{metric}' as not in dataframe.")
                continue

            metric_values = all_metrics[
                (all_metrics["attribute_name"] == attribute_name)
                & (all_metrics["attribute_value"] == attribute_value)
            ][metric].dropna()

            disparity_stats[metric] = {
                "mean": metric_values.mean(),
                "std": metric_values.std(),
                "min": metric_values.min(),
                "25%": metric_values.quantile(0.25),
                "50%": metric_values.median(),
                "75%": metric_values.quantile(0.75),
                "max": metric_values.max(),
            }

        return disparity_stats

    # Select the appropriate bootstrap method
    if bootstrap_method == "stratified":
        bootstrapped_samples = stratified_bootstrap(
            df, seeds, n_iterations, sample_size, stratify_columns
        )
    elif bootstrap_method == "balanced":
        bootstrapped_samples = balanced_bootstrap(
            df, seeds, n_iterations, sample_size, stratify_columns
        )
    else:
        raise ValueError("Invalid bootstrap method. Choose 'stratified' or 'balanced'.")

    # Run aequitas audit on each bootstrapped sample and collect results
    metrics_results = []

    for sample in tqdm(bootstrapped_samples, desc="Running audits"):
        # Perform audit
        audit_df = sample[categorical_columns + [score_column, label_column]]
        audit = Audit(
            df=audit_df,
            score_column=score_column,
            label_column=label_column,
        )
        audit.audit()
        metrics_results.append(audit.disparity_df)

    # Concatenate all metrics results
    all_metrics = pd.concat(metrics_results)

    # Reset index to access 'attribute_name' and 'attribute_value' as columns
    all_metrics.reset_index(inplace=True)

    # Calculate disparity statistics if requested
    if calculate_statistics and attribute_name and attribute_value:
        disparity_stats = calculate_disparity_statistics(
            all_metrics, attribute_name, attribute_value
        )
    else:
        disparity_stats = None

    # Handle disparity metrics
    if return_disparity_metrics:
        all_metrics_dispar = all_metrics[
            [col for col in all_metrics.columns if "disparity" in col]
            + ["attribute_name", "attribute_value"]
        ]
        results_dict["all_metrics_dispar"] = all_metrics_dispar

    results_dict["all_metrics"] = all_metrics
    results_dict["disparity_stats"] = disparity_stats

    return results_dict


################################################################################
######################### Plot Disparity Distributions #########################
################################################################################

## TODO L.S.
## make ref_group always straight line regardless of ref_group


def plot_metrics(
    df,
    metric_cols,
    categories="all",
    include_legend=True,
    cmap="tab20c",
    save_plots=False,
    image_path_png=None,
    image_path_svg=None,
):
    """
    Plots violin plots for specified metric columns grouped by attribute names
    and optionally saves the output.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be plotted.
    It should include 'attribute_name' and 'attribute_value' columns.
    metric_cols (list of str): List of metric column names to plot.
    categories (str or list of str): 'all' to plot all categories, a specific
    category as a string, or a list of specific attribute names to plot.
    include_legend (bool): Whether to include the legend in the plots.
    Default is True.
    cmap (str): The color map to use for the plots. Default is 'tab20c'.
    save_plots (bool): Whether to save the plots to the specified paths.
    Default is False.
    image_path_png (str): Optional path to save the plots as PNG files.
    Default is None.
    image_path_svg (str): Optional path to save the plots as SVG files.
    Default is None.

    Raises:
    KeyError: If the required columns 'attribute_name' or 'attribute_value' are
    not present in the DataFrame.
    ValueError: If categories is not 'all', a string, or a list of specific
    attribute names.

    Example usage:
    metric_cols = ["pprev_disparity", "fpr_disparity", "tnr_disparity",
                   "tpr_disparity", "fnr_disparity", "precision_disparity"]
    plot_metrics(
        all_metrics_stratified,
        categories="all",
        metric_cols=metric_cols,
        include_legend=True,
        cmap="tab20c",
        save_plots=True,
        image_path_png="./plots",
        image_path_svg="./plots",
    )
    """

    # Ensure necessary columns are in the DataFrame
    if "attribute_name" not in df.columns or "attribute_value" not in df.columns:
        raise KeyError(
            "The DataFrame must contain 'attribute_name' and 'attribute_value' columns."
        )

    # Filter the DataFrame based on the specified categories
    if categories != "all":
        if isinstance(categories, str):
            categories = [categories]
        elif not isinstance(categories, list):
            raise ValueError(
                "categories should be 'all', a string, or a list of specific attribute names."
            )
        df = df[df["attribute_name"].isin(categories)]

    for name, rows in df.groupby("attribute_name"):
        # Create a dictionary to map attribute_value to labels A, B, C, etc.
        value_labels = {
            value: chr(65 + i)
            for i, value in enumerate(rows["attribute_value"].unique())
        }

        # Reverse the dictionary to use in plotting
        label_values = {v: k for k, v in value_labels.items()}

        # Use a color map to generate colors
        color_map = plt.get_cmap(cmap)  # Allow user to specify colormap
        num_colors = len(label_values)
        colors = [color_map(i / num_colors) for i in range(num_colors)]

        # Create custom legend handles
        legend_handles = [
            plt.Line2D([0], [0], color=colors[j], lw=4, label=f"{label} = {value}")
            for j, (label, value) in enumerate(label_values.items())
        ]

        n_metrics = len(metric_cols)
        n_cols = 6
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(24, 4), squeeze=False)
        rows["label"] = rows["attribute_value"].map(value_labels)

        for i, col in enumerate(metric_cols):
            ax = axs[i // n_cols, i % n_cols]
            sns.violinplot(ax=ax, x="label", y=col, data=rows)
            ax.set_title(name + "_" + col)
            ax.set_xlabel("")
            ax.set_xticks(range(len(label_values)))
            ax.set_xticklabels(
                label_values.keys(),
                rotation=0,
                fontweight="bold",
            )

            # Set the color and font weight of each tick label to match the
            # corresponding color in the legend
            for tick_label in ax.get_xticklabels():
                tick_label.set_color(
                    colors[list(label_values.keys()).index(tick_label.get_text())]
                )
                tick_label.set_fontweight("bold")

            ax.hlines(0, -1, len(value_labels) + 1, ls=":", color="red")
            ax.hlines(1, -1, len(value_labels) + 1, ls=":")
            ax.hlines(2, -1, len(value_labels) + 1, ls=":", color="red")
            ax.set_xlim([-1, len(value_labels)])
            ax.set_ylim(-2, 4)

        # Remove any empty subplots
        for j in range(i + 1, n_rows * n_cols):
            fig.delaxes(axs[j // n_cols, j % n_cols])

        # Add a single legend at the top of the figure if include_legend is True
        if include_legend:
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                ncol=len(label_values),
                bbox_to_anchor=(0.5, 1.15),
                fontsize="large",
            )

        plt.tight_layout(
            w_pad=2, h_pad=2, rect=[0.01, 0.01, 1.01, 1]
        )  # Adjust rect to make space for the legend and reduce white space

        # Save plots if save_plots is True and paths are provided
        if save_plots:
            if image_path_png:
                os.makedirs(
                    image_path_png, exist_ok=True
                )  # Ensure PNG directory exists
                png_file = os.path.join(image_path_png, f"{name}_plot.png")
                plt.savefig(png_file, bbox_inches="tight")
            if image_path_svg:
                os.makedirs(
                    image_path_svg, exist_ok=True
                )  # Ensure SVG directory exists
                svg_file = os.path.join(image_path_svg, f"{name}_plot.svg")
                plt.savefig(svg_file, bbox_inches="tight")

        plt.show()


def plot_mean_std_disparity(data_list, years=None):
    """
    Plots the mean disparity scores over years with error bars representing the standard deviation.

    Parameters:
    - data_list: List of dictionaries. Each dictionary represents a year's data
                 with race categories as keys and disparity scores as values.
    - years: Optional list of years corresponding to each dictionary in data_list.
             If not provided, years will be assigned sequentially starting from 2020.
    """
    # Initialize an empty DataFrame to store mean and std
    stats_df = pd.DataFrame(columns=["Year", "Race", "MeanDisparity", "StdDisparity"])

    # If years are not provided, assign default years starting from 2020
    if years is None:
        years = [2020 - len(data_list) + i + 1 for i in range(len(data_list))]
    elif len(years) != len(data_list):
        raise ValueError("Length of years list must match length of data_list.")

    # Iterate over the data_list and years to compute mean and std
    for data, year in zip(data_list, years):
        for race, scores in data.items():
            # Convert scores to numpy array if it's a pandas Series
            if isinstance(scores, (pd.Series, pd.DataFrame)):
                scores = scores.values.flatten()
            elif isinstance(scores, list):
                scores = np.array(scores)
            else:
                scores = np.array([scores])
            # Compute mean and std deviation
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            # Append to stats_df
            temp_df = pd.DataFrame(
                {
                    "Year": [year],
                    "Race": [race],
                    "MeanDisparity": [mean_score],
                    "StdDisparity": [std_score],
                }
            )
            stats_df = pd.concat([stats_df, temp_df], ignore_index=True)

    # Convert 'Year' to numeric for proper plotting
    stats_df["Year"] = pd.to_numeric(stats_df["Year"])

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Get the list of races
    races = stats_df["Race"].unique()

    # Define a color palette
    colors = plt.cm.tab10.colors  # Adjust the colormap as needed
    color_dict = {race: colors[i % len(colors)] for i, race in enumerate(races)}

    # Plot mean disparity values over years for each race with error bars
    for race in races:
        race_data = stats_df[stats_df["Race"] == race].sort_values("Year")
        plt.errorbar(
            race_data["Year"],
            race_data["MeanDisparity"],
            yerr=race_data["StdDisparity"],
            marker="o",
            linestyle="-",
            color=color_dict[race],
            label=race,
            capsize=5,
        )

    # Add horizontal dotted lines at y=0 (red), y=1 (blue), and y=2 (red)
    plt.axhline(y=0, linestyle="--", color="red")
    plt.axhline(y=1, linestyle="--", color="blue")
    plt.axhline(y=2, linestyle="--", color="red")

    # Set plot titles and labels
    plt.title("Mean Disparity Scores Over Years with Standard Deviation")
    plt.ylabel("Mean Disparity Score")
    plt.xlabel("Year")

    # Set x-axis ticks to integer years
    plt.xticks(years)  # Ensure x-axis ticks are at the specified years
    plt.xlim(min(years) - 0.5, max(years) + 0.5)  # Adjust x-axis limits

    # Format x-axis to show integer labels
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    # Add legend
    plt.legend(title="Race")

    # Show grid
    plt.grid(True, linestyle="--", alpha=0.5)

    # Adjust layout to prevent clipping of labels and legend
    plt.tight_layout()

    # Show the plot
    plt.show()


################################################################################


def plot_metrics_with_ks_test(
    df,
    metric_cols,
    pass_rate,
    categories="all",
    include_legend=True,
    cmap="tab20c",
    significance_level=0.05,
):
    """
    Plots violin plots for specified metric columns grouped by attribute names
    and annotates with KS test results.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be plotted.
    metric_cols (list of str): List of metric column names to plot.
    ks_test_results_df (pd.DataFrame): DataFrame containing the KS test results.
    categories (str or list of str): 'all' to plot all categories or a specific
    category.
    include_legend (bool): Whether to include the legend in the plots.
    Default is True.
    cmap (str): The color map to use for the plots. Default is 'tab20c'.
    significance_level (float): The significance level for the KS test
    Default is 0.05.
    """

    # Ensure necessary columns are in the DataFrame
    if "attribute_name" not in df.columns or "attribute_value" not in df.columns:
        raise KeyError(
            "The DataFrame must contain 'attribute_name' and 'attribute_value' columns."
        )

    # Filter the DataFrame based on the specified categories
    if categories != "all":
        if isinstance(categories, str):
            categories = [categories]
        elif not isinstance(categories, list):
            raise ValueError(
                "categories should be 'all', a string, or a list of specific attribute names."
            )
        df = df[df["attribute_name"].isin(categories)]

    for name, rows in df.groupby("attribute_name"):
        value_labels = {
            value: chr(65 + i)
            for i, value in enumerate(rows["attribute_value"].unique())
        }

        label_values = {v: k for k, v in value_labels.items()}

        color_map = plt.get_cmap(cmap)
        num_colors = len(label_values)
        colors = [color_map(i / num_colors) for i in range(num_colors)]

        legend_handles = [
            plt.Line2D([0], [0], color=colors[j], lw=4, label=f"{label} = {value}")
            for j, (label, value) in enumerate(label_values.items())
        ]

        n_metrics = len(metric_cols)
        n_cols = 6
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(24, 4), squeeze=False)
        rows["label"] = rows["attribute_value"].map(value_labels)

        for i, col in enumerate(metric_cols):
            ax = axs[i // n_cols, i % n_cols]
            sns.violinplot(ax=ax, x="label", y=col, data=rows)
            cur_rate = pass_rate.loc[name, col.replace("_disparity", "")]

            ax.set_title(name + "_" + col)
            ax.set_suptitle("pass", color="green")
            ax.set_xlabel("")
            ax.set_xticks(range(len(label_values)))
            ax.set_xticklabels(
                label_values.keys(),
                rotation=0,
                fontweight="bold",
            )

            for tick_label in ax.get_xticklabels():
                tick_label.set_color(
                    colors[list(label_values.keys()).index(tick_label.get_text())]
                )
                tick_label.set_fontweight("bold")

            ax.hlines(0, -1, len(value_labels) + 1, ls=":", color="red")
            ax.hlines(1, -1, len(value_labels) + 1, ls=":")
            ax.hlines(2, -1, len(value_labels) + 1, ls=":", color="red")
            ax.set_xlim([-1, len(value_labels)])
            ax.set_ylim(-2, 4)

        for j in range(i + 1, n_rows * n_cols):
            fig.delaxes(axs[j // n_cols, j % n_cols])

        if include_legend:
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                ncol=len(label_values),
                bbox_to_anchor=(0.5, 1.15),
                fontsize="large",
            )

        plt.tight_layout(w_pad=2, h_pad=2, rect=[0.01, 0.01, 1.01, 1])
        plt.show()
