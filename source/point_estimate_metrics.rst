.. _point_estimates:   

.. _target-link:


Point Estimate Evaluation
==========================================

EquiBoots includes an interface for computing group-specific and overall 
point estimates for key performance metrics across classification and regression 
tasks. These estimates serve as the foundation for fairness auditing by quantifying 
how models perform across different demographic groups or other sensitive attributes.

The EquiBoots class allows you to generate point estimates of model performance 
metrics across demographic groups using your specified fairness variables. These 
estimates provide a transparent snapshot of how your model performs across different 
subpopulations.

This section shows how to compute group-specific metrics without bootstrapping. 
For bootstrapped metrics and confidence intervals, refer to the 
:ref:`bootstrapped metrics evaluation section <Bootstrapped_Metrics>`.

Supported Metrics
-------------------

For classification tasks (binary, multi-class, and multi-label), the following metrics are supported:

- Accuracy, Precision, Recall, F1-score

- AUROC, AUPRC (for probabilistic models)

- Calibration Area Under The Curve 

- Log Loss, Brier Score

For regression tasks:

- :math:`R^2`, MAE, MSE, RMSE

- Group-based residual plots

Implementation
-------------------

Point estimates are computed using the ``EquiBoots`` class and its ``grouper`` method:

.. code:: python 

    # Generate synthetic test data
    y_prob = np.random.rand(1000)
    y_pred = y_prob > 0.5
    y_true = np.random.randint(0, 2, 1000)

    race = (
        np.random.RandomState(3)
        .choice(["white", "black", "asian", "hispanic"], 1000)
        .reshape(-1, 1)
    )
    sex = np.random.choice(["M", "F"], 1000).reshape(-1, 1)

    fairness_df = pd.DataFrame(
        data=np.concatenate((race, sex), axis=1), columns=["race", "sex"]
    )

    # Initialize and process groups
    eq = eqb.EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
    )
    eq.grouper(groupings_vars=["race", "sex"])
    sliced_data = eq.slicer("race")