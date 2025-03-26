[![Downloads](https://pepy.tech/badge/model_tuner)](https://pepy.tech/project/model_tuner) [![PyPI](https://img.shields.io/pypi/v/model_tuner.svg)](https://pypi.org/project/model_tuner/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12727322.svg)](https://doi.org/10.5281/zenodo.12727322)

The `equiboots` library is a fairness-aware model evaluation toolkit designed to audit performance disparities across demographic groups. It provides robust, bootstrapped metrics for binary, multi-class, and multi-label classification, as well as regression models. The library supports group-wise performance slicing, fairness diagnostics, and customizable visualizations to support equitable AI/ML development.

`equiboots` is particularly useful in clinical, social, and policy domains where transparency, bias mitigation, and outcome fairness are critical for responsible deployment.

## Prerequisites

Before installing `equiboots`, ensure your system meets the following requirements:

## Python Version

`equiboots` requires **Python 3.7.4 or higher**. Specific dependency versions vary depending on your Python version.

## Dependencies

The following dependencies will be automatically installed with `equiboots`:

`matplotlib>=3.5.3, <=3.10.1`
`numpy>=1.21.6, <=2.2.4`
`pandas>=1.3.5, <=2.2.3`
`scikit-learn>=1.0.2, <=1.5.2`
`scipy>=1.8.0, <=1.15.2`
`seaborn>=0.11.2, <=0.13.2`
`tqdm>=4.66.4, <=4.67.1`
`
## 💾 Installation

You can install `equiboots` directly from PyPI:

```bash
pip install equiboots
```

## 📄 Official Documentation

Coming soon.


## 🌐 Author Website

https://www.mii.ucla.edu/

## ⚖️ License

`model_tuner` is distributed under the Apache License. See [LICENSE](https://github.com/uclamii/model_tuner?tab=Apache-2.0-1-ov-file) for more information.

## 📚 Citing `model_tuner`

If you use `model_tuner` in your research or projects, please consider citing it.

```bibtex
@software{funnell_2024_12727322,
  author       = {Funnell, Arthur and
                  Shpaner, Leonid and
                  Petousis, Panayiotis},
  title        = {Model Tuner},
  month        = jul,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.0.28b},
  doi          = {10.5281/zenodo.12727322},
  url          = {https://doi.org/10.5281/zenodo.12727322}
}
```

## Support

If you have any questions or issues with `model_tuner`, please open an issue on this [GitHub repository](https://github.com/uclamii/model_tuner/).

## Acknowledgements

This work was supported by the UCLA Medical Informatics Institute (MII) and the Clinical and Translational Science Institute (CTSI). Special thanks to Dr. Alex Bui for his invaluable guidance and support, and to Panayiotis Petousis for his original contributions to this codebase.