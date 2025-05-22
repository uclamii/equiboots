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
`pingouin>=0.5.3`  

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
  version      = {0.0.0a4},
  doi          = {10.5281/zenodo.15086941},
  url          = {https://doi.org/10.5281/zenodo.15086941}
}
```

## Support

If you have any questions or issues with `equiboots`, please open an issue on this [GitHub repository](https://github.com/uclamii/equiboots/).

## Acknowledgements

This work was supported by the UCLA Medical Informatics Institute (MII) and the Clinical and Translational Science Institute (CTSI). Special thanks to Dr. Alex Bui for his invaluable guidance and support, and to Panayiotis Petousis, PhD, for his contributions to this codebase.

