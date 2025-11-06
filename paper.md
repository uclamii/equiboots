---
title: "EquiBoots: A Python Library for Fairness-Aware Model Evaluation, Statistical Testing, and Model Healing"
tags:
  - Python
  - fairness
  - machine learning
  - bias mitigation
  - uncertainty estimation
  - open source
authors:
  - name: Leonid Shpaner
    orcid: 0009-0007-5311-8095
    equal-contrib: true
    affiliation: 1
  - name: Arthur Funnell
    orcid: 0009-0002-6423-861X
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: UCLA Health, Los Angeles, United States
   index: 1
date: 6 November 2025
bibliography: paper.bib
---

# Summary

**EquiBoots** is an open-source Python library that enables fairness-aware model evaluation through bootstrapped statistical testing, bias quantification, and model threshold correction (healing).  
It identifies disparities across subgroups, performs hypothesis testing, visualizes group performance, and provides automated routines to rebalance thresholds for improved equity.  

Designed for seamless integration with scikit-learn pipelines, EquiBoots bridges the gap between fairness research and real-world deployment in applied machine learning.

You can install the package directly from PyPI:

```text
pip install equiboots
```

Source code: https://github.com/uclamii/equi_boots

Documentation: https://lshpaner.github.io/equiboots_docs

Tutorial: https://igit.me/equi

DOI: https://doi.org/10.5281/zenodo.14002139


## Statement of need

Machine learning models often exhibit disparities in performance across demographic or clinical groups due to unbalanced datasets or biased thresholds.
Bias can stem from unrepresentative training data (data bias) or from algorithms that amplify inequities in model predictions (algorithmic bias).

In healthcare, for instance:

- Algorithms may under-prioritize Black patients for care management. 

- Dermatology models may perform poorly on darker skin tones due to limited dataset diversity.


EquiBoots was developed to make these disparities quantifiable and correctable.
It provides transparent, statistically grounded tools for fairness auditing that help researchers and practitioners detect, measure, and mitigate bias in predictive models.

## Features
### Grouped Evaluation and Statistical Testing

EquiBoots allows users to slice model predictions by sensitive attributes such as race or sex and compute group-specific metrics including AUC, precision, recall, and calibration.
It includes built-in statistical testing modules (e.g., t-tests, Mann–Whitney U tests, bootstrapped differences) to determine whether disparities are statistically significant.

Results can be visualized through grouped Receiver Operating Characteristic (ROC), Precision-Recall (PR) curves, and calibration curves, enabling interpretable fairness assessments.

Figure 1. Grouped ROC AUC and significance testing across demographic subgroups.