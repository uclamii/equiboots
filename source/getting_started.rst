.. _getting_started:   

.. _target-link:


Welcome to the EquiBoots Documentation!
==========================================

.. note::
   This documentation is for ``equiboots`` version ``0.0.0a3``.

EquiBoots is a fairness-aware model evaluation toolkit for auditing performance 
disparities across demographic groups in machine learning models. It provides robust, 
bootstrapped evaluation metrics for binary, multi-class, and multi-label classification 
tasks, as well as regression models.

The library supports:

- Group-wise performance slicing
- Fairness diagnostics and disparity metrics
- Confidence intervals via bootstrapping
- Customizable and publication-ready visualizations

EquiBoots is particularly suited for applications in clinical, social, and 
policy contexts—domains where transparency, bias mitigation, and equitable outcomes 
are essential for responsible AI/ML deployment.

Project Links
---------------

1. `PyPI Page <https://pypi.org/project/equiboots/>`_  

2. `GitHub Repository <https://github.com/uclamii/equiboots>`_

Prerequisites
-------------
Before you install ``equiboots``, ensure your system meets the following requirements:

- **Python** (version ``3.7.4`` or higher)

Additionally, ``equiboots`` depends on the following packages, which will be automatically installed when you install ``equiboots``:

- ``matplotlib``: version ``3.5.3`` or higher, but capped at ``3.10.1``
- ``numpy``: version ``1.21.6`` or higher, but capped at ``2.2.4``
- ``pandas``: version ``1.3.5`` or higher, but capped at ``2.2.3``
- ``scikit-learn``: version ``1.0.2`` or higher, but capped at ``1.5.2``
- ``scipy``: version ``1.8.0`` or higher, but capped at ``1.15.2``
- ``seaborn``: version ``0.11.2`` or higher, but capped at ``0.13.2``
- ``tqdm```: version ``4.66.4`` or higher, but capped below ``4.67.1``

.. _installation:

Installation
-------------

You can install ``equiboots`` directly from PyPI:

.. code-block:: python

    pip install equiboots

Description
--------------

This guide provides detailed instructions and examples for using the functions 
provided in the ``equiboots`` library and how to use them effectively in your projects.
