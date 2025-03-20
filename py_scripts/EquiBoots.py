import pandas as pd
import numpy as np

class EquiBoots:
    
    def __init__(self, y_prob: np.array, y_true: np.array, fairness_df: pd.DataFrame, fariness_vars: list, task: str ='binary_classification') -> None:
        
        ## Exceptions
        self.check_task(task)
        self.check_fairness_vars(fariness_vars)
        
        pass
    
    def check_task(self, task):
        if task not in ['binary_classifcation', 'multi_class_classification', 'regression', 'multi_label_classification']:
            raise ValueError("Invalid task, please supply one of 'binary_classification', 'multi_class_classification', 'regression' or 'multi_label_classification'")
        
    def check_fairness_vars(self, fairness_vars):
        if fairness_vars is None:
            raise ValueError("fairness_vars cannot be None, please provide a list")
        if not isinstance(fairness_vars, list):
            raise ValueError("fairness_vars must be a list")
