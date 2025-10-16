<img src="https://raw.githubusercontent.com/uclamii/equiboots/refs/heads/main/logo/EquiBoots.png" width="300" style="border: none; outline: none; box-shadow: none;" oncontextmenu="return false;">

<br>

[![Downloads](https://pepy.tech/badge/equiboots)](https://pepy.tech/project/equiboots) [![PyPI](https://img.shields.io/pypi/v/equiboots.svg)](https://pypi.org/project/equiboots/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15086941.svg)](https://doi.org/10.5281/zenodo.15086941)


The `equiboots` library is a fairness-aware model evaluation toolkit designed to audit performance disparities across demographic groups. It provides robust, bootstrapped metrics for binary, multi-class, and multi-label classification, as well as regression models. The library supports group-wise performance slicing, fairness diagnostics, and customizable visualizations to support equitable AI/ML development.

`equiboots` is particularly useful in clinical, social, and policy domains where transparency, bias mitigation, and outcome fairness are critical for responsible deployment.

## Prerequisites

Before installing `equiboots`, ensure your system meets the following requirements:

## Python Version

`equiboots` requires **Python 3.7.4 or higher**. Specific dependency versions vary depending on your Python version.

## Dependencies

The following dependencies will be automatically installed with `equiboots`:

- `matplotlib>=3.5.3, <=3.10.1`
- `numpy>=1.21.6, <=2.2.4`
- `pandas>=1.3.5, <=2.2.3`
- `scikit-learn>=1.0.2, <=1.5.2`
- `scipy>=1.8.0, <=1.15.2`
- `seaborn>=0.11.2, <=0.13.2`
- `statsmodels>=0.13, <=0.14.4`
- `tqdm>=4.66.4, <=4.67.1`

## 💾 Installation

You can install `equiboots` directly from PyPI:

```bash
pip install equiboots
```

## 📄 Official Documentation

https://uclamii.github.io/equiboots_docs

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
                   Beam, Colin and
                   Petousis, Panayiotis},
   title        = {EquiBoots},
   month        = mar,
   year         = 2025,
   publisher    = {Zenodo},
   version      = {0.0.1a7},
   doi          = {10.5281/zenodo.15086941},
   url          = {https://doi.org/10.5281/zenodo.15086941}
}
```

## Support

If you have any questions or issues with `equiboots`, please open an issue on this [GitHub repository](https://github.com/uclamii/equiboots/).

## Acknowledgements

This work was supported by the UCLA Medical Informatics Institute (MII) and the Clinical and Translational Science Institute (CTSI). Special thanks to Alex Bui, PhD, for his invaluable guidance and support. Many thanks to David Elashoff, PhD, and Sitaram Vangala, M.S., for their statistical consultation. Thanks to Jayleen Mendoza for her contribution to model healing.

