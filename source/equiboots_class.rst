.. _equiboots_class:   

.. _target-link:

EquiBoots Class
===============

.. automodule:: equiboots.EquiBootsClass
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``EquiBoots`` class provides tools for fairness-aware evaluation and bootstrapping of machine learning model predictions. It supports binary, multi-class, multi-label classification, and regression tasks, and enables group-based metric calculation, disparity analysis, and statistical significance testing.

Constructor
-----------

.. py:class:: EquiBoots(y_true, y_pred, fairness_df, fairness_vars, y_prob=None, seeds=[1,2,3,4,5,6,7,8,9,10], reference_groups=None, task="binary_classification", bootstrap_flag=False, num_bootstraps=10, boot_sample_size=100, balanced=True, stratify_by_outcome=False, group_min_size=10)

   :param y_true: Ground truth labels (numpy array)
   :param y_pred: Predicted labels (numpy array)
   :param fairness_df: DataFrame with fairness variables (pandas DataFrame)
   :param fairness_vars: List of fairness variable names (list)
   :param y_prob: Predicted probabilities (numpy array, optional)
   :param seeds: Random seeds for bootstrapping (list, optional)
   :param reference_groups: Reference group for each fairness variable (list, optional)
   :param task: Task type ("binary_classification", "multi_class_classification", "regression", "multi_label_classification")
   :param bootstrap_flag: Whether to use bootstrapping (bool)
   :param num_bootstraps: Number of bootstrap iterations (int)
   :param boot_sample_size: Size of each bootstrap sample (int)
   :param balanced: Whether to balance samples across groups (bool), if False groups are stratified (proportions of groups from original dataset is maintaned) if True groups are balanced (equal number of samples from each group)
   :param stratify_by_outcome: Stratify sampling by outcome (bool)
   :param group_min_size: Any group with less observation than Minimum group size (int) gets ommitted

Main Methods
------------

.. py:method:: grouper(groupings_vars)
   :param groupings_vars: List of variables to group by.
   :returns: None

   Groups data by the specified fairness variables and stores indices for each category.

.. py:method:: slicer(slicing_var)
   :param slicing_var: The variable to slice by.
   :returns: Dictionary or list of dictionaries with grouped data.

   Slices y_true, y_prob, and y_pred by the specified variable, with or without bootstrapping.

.. py:method:: get_metrics(sliced_dict)
   :param sliced_dict: Output from slicer.
   :returns: Metrics for each group (dict or list of dicts).

   Calculates metrics for each group based on the task type.

.. py:method:: calculate_disparities(metric_dict, var_name)
   :param metric_dict: Metrics for each group.
   :param var_name: Fairness variable name.
   :returns: Disparities for each group (dict or list of dicts).

   Calculates ratio disparities between each group and the reference group.

.. py:method:: calculate_differences(metric_dict, ref_var_name)
   :param metric_dict: Metrics for each group.
   :param ref_var_name: Fairness variable name.
   :returns: Differences for each group (dict or list of dicts).

   Calculates difference disparities between each group and the reference group.

.. py:method:: analyze_statistical_significance(metric_dict, var_name, test_config, differences=None)
   :param metric_dict: Metrics for each group.
   :param var_name: Fairness variable name.
   :param test_config: Statistical test configuration dictionary.
   :param differences: Optional precomputed differences.
   :returns: Dictionary of statistical test results.

   Analyzes statistical significance of metric differences between groups.

.. py:method:: set_fix_seeds(seeds)
   :param seeds: List of integer seeds.
   :returns: None

   Sets fixed random seeds for reproducibility.

.. py:method:: list_available_tests()
   :returns: Dictionary of available statistical tests.

.. py:method:: list_adjustment_methods()
   :returns: Dictionary of available adjustment methods.

Non-Main/Internal Methods
-------------------------

These methods are primarily used internally by the class for validation, group management, and sampling, but may be useful for advanced users:

.. py:method:: set_reference_groups(reference_groups)
   Sets the reference group for each fairness variable, either from user input or by selecting the most populous group.

.. py:method:: check_task(task)
   Validates the task type.

.. py:method:: check_classification_task(task)
   Ensures the task is a classification task when required.

.. py:method:: check_fairness_vars(fairness_vars)
   Validates the fairness_vars input.

.. py:method:: check_group_size(group, cat, var)
   Checks if a group meets the minimum size requirement.

.. py:method:: check_group_empty(sampled_group, cat, var)
   Checks if a sampled group is empty.

.. py:method:: sample_group(group, n_categories, indx, sample_size, seeds, balanced)
   Samples a group with or without balancing.

.. py:method:: groups_slicer(groups, slicing_var)
   Slices y_true, y_prob, and y_pred into categories of a given variable.

.. py:method:: get_groups_metrics(sliced_dict)
   Calculates metrics for each group based on the task type.

.. py:method:: calculate_groups_disparities(metric_dict, var_name)
   Calculates disparities between each group and the reference group.

.. py:method:: calculate_groups_differences(metric_dict, ref_var_name)
   Calculates differences between each group and the reference group.

Example Usage
=============

Below are two dummy examples demonstrating how to use the ``EquiBoots`` class: one **without** bootstrapping and one **with** bootstrapping.

For more detailed examples, refer to this colab notebook or
the py_scripts/testingscript.py.

Point Estimates Without Bootstrapping
---------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from equiboots import EquiBoots

    # Example data
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.7, 0.4, 0.9])
    y_pred = np.array([0, 1, 1, 0, 1])
    fairness_df = pd.DataFrame({
        "race": ["A", "B", "A", "B", "A"],
        "sex": ["M", "F", "F", "M", "F"]
    })

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
        task="binary_classification",
        bootstrap_flag=False
    )

    eq.grouper(groupings_vars=["race"])
    sliced = eq.slicer("race")
    metrics = eq.get_metrics(sliced)
    disparities = eq.calculate_disparities(metrics, "race")

    print("Metrics by group:", metrics)
    print("Disparities:", disparities)

With Bootstrapping
------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from equiboots import EquiBoots

    # Example data
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.7, 0.4, 0.9])
    y_pred = np.array([0, 1, 1, 0, 1])
    fairness_df = pd.DataFrame({
        "race": ["A", "B", "A", "B", "A"],
        "sex": ["M", "F", "F", "M", "F"]
    })

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
        task="binary_classification",
        bootstrap_flag=True,
        num_bootstraps=5,
        boot_sample_size=5
    )

    eq.grouper(groupings_vars=["race"])
    sliced = eq.slicer("race")
    metrics = eq.get_metrics(sliced)
    disparities = eq.calculate_disparities(metrics, "race")

    print("Metrics by group (bootstrapped):", metrics)
    print("Disparities (bootstrapped):", disparities)
