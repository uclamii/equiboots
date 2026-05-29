import pandas as pd
import numpy as np


def metrics_table(
    metrics,
    statistical_tests=None,
    differences=None,
    reference_group=None,
    decimal_places=3,
):

    ### check if group_differences is a string
    ### if it is a string then this is a bootstrap table

    if isinstance(differences, list):

        mean_differences = {}

        all_groups = set()
        for bootstrap_sample in differences:
            all_groups.update(bootstrap_sample.keys())

        for group in all_groups:
            if group == reference_group:
                continue

            group_means = {}

            # Get all metrics from first bootstrap sample
            if differences and group in differences[0]:
                metrics_list = list(differences[0][group].keys())

                # Calculate mean for each metric across all bootstraps
                for metric in metrics_list:
                    values = []
                    for bootstrap_sample in differences:
                        if (
                            group in bootstrap_sample
                            and metric in bootstrap_sample[group]
                        ):
                            values.append(bootstrap_sample[group][metric])

                    if values:
                        mean_value = np.mean(values)
                        if statistical_tests:
                            if (
                                group in statistical_tests
                                and metric in statistical_tests[group]
                                and statistical_tests[group][metric].is_significant
                            ):
                                group_means[metric] = (
                                    f"{mean_value:.{decimal_places}f} *"
                                )
                            else:
                                group_means[metric] = f"{mean_value:.{decimal_places}f}"
                        else:
                            group_means[metric] = mean_value
                    else:
                        group_means[metric] = np.nan

            mean_differences[group] = group_means

        metrics_table = pd.DataFrame(mean_differences)
        return metrics_table
    else:

        metrics_table = pd.DataFrame(metrics)

        if statistical_tests:
            # statistical_tests is now nested: {outer: {metric: StatTestResult}}
            omnibus_dict = statistical_tests.get("omnibus", {})

            # Star ALL columns if any omnibus metric is significant
            if isinstance(omnibus_dict, dict) and any(
                t.is_significant for t in omnibus_dict.values()
            ):
                metrics_table.columns = [f"{col} *" for col in metrics_table.columns]

            # Triangle a specific group's column if any of its pairwise metrics is significant
            for test_name, test_dict in statistical_tests.items():
                if test_name == "omnibus" or not isinstance(test_dict, dict):
                    continue
                if any(t.is_significant for t in test_dict.values()):
                    metrics_table.columns = [
                        f"{col} ▲" if test_name in col else col
                        for col in metrics_table.columns
                    ]

            ### Dropping irrelevant columns if doing statistical tests
            metrics_table = metrics_table.drop(
                index=[
                    "Brier Score",
                    "Log Loss",
                    "Average Precision Score",
                    "ROC AUC",
                    "Prevalence",
                    "Calibration AUC",
                ],
                errors="ignore",
            )

        metrics_table.round(3)

        return metrics_table
