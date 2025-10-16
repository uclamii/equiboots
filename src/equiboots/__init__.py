from .EquiBootsClass import *
from .metrics import *
from .plots import *
from .healer import *
from .logo import *

import sys
import builtins

# Detailed Documentation

detailed_doc = """
The `equiboots` library is a fairness-aware model evaluation toolkit designed to
audit performance disparities across demographic groups. It provides robust, 
bootstrapped metrics for binary, multi-class, and multi-label classification, 
as well as regression models. The library supports group-wise performance slicing, 
fairness diagnostics, and customizable visualizations to support equitable AI/ML 
development.

EquiBoots is particularly useful in clinical, social, and policy domains where 
transparency, bias mitigation, and outcome fairness are critical for responsible 
deployment.

PyPI: https://pypi.org/project/equiboots  
Documentation: https://uclamii.github.io/equiboots_docs/

Version: 0.0.1a7
"""


# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc


__version__ = "0.0.1a7"
__author__ = (
    "Leonid Shpaner, Arthur Funnell, Al Rahrooh, Colin Beam, Panayiotis Petousis"
)
__email__ = "lshpaner@ucla.edu; alafunnell@gmail.com; arahrooh@ucla.edu; CBeam@mednet.ucla.edu; pp89@ucla.edu"


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
