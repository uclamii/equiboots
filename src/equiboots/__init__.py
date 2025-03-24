from .EquiBoots import *
from .metrics import *
from .plots import *
from .logo import *

import sys
import builtins

# Detailed Documentation

detailed_doc = """                                                               
The `model_tuner` library is a versatile and powerful tool designed to 
facilitate the training, tuning, and evaluation of machine learning models. 
It supports various functionalities such as handling imbalanced data, applying 
different scaling and imputation techniques, calibrating models, and conducting 
cross-validation. This library is particularly useful for hyperparameter tuning
and ensuring optimal performance across different metrics.

PyPI: https://pypi.org/project/equiboots
Documentation: https://uclamii.github.io/equiboots/


Version: 0.0.0a

"""

# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc


__version__ = "0.0.0a"
__author__ = "Leonid Shpaner, Arthur Funnell, Panayiotis Petousis"
__email__ = "lshpaner@ucla.edu; alafunnell@gmail.com; pp89@ucla.edu"


# Define the custom help function
def custom_help(obj=None):
    """
    Custom help function to dynamically include ASCII art in help() output.
    """
    if (
        obj is None or obj is sys.modules[__name__]
    ):  # When `help()` is called for this module
        print(equiboots_logo)  # Print ASCII art first
        print(detailed_doc)  # Print the detailed documentation
    else:
        original_help(obj)  # Use the original help for other objects


# Backup the original help function
original_help = builtins.help

# Override the global help function in builtins
builtins.help = custom_help
