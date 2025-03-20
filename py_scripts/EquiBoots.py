import pandas as pd
import numpy as np

class EquiBoots:
    
    def __init__(self, y_prob: np.array, y_true: np.array, fairness_df: pd.DataFrame, task: str ='binary_classification', fairness_vars: list=[None],) -> None:
        
        self.fairness_vars = fairness_vars
        self.task = task
        self.y_prob = y_prob
        self.y_true = y_true
        self.fairness_df = fairness_df
        self.groups = {}
        pass


    def grouper(self, groupings_vars: list) -> pd.DataFrame:
        """Method that given a list of categorical variables, returns indices of each category."""
        for var in groupings_vars:
            self.groups[var] = {}
            # Replace NaN with 'missing' to treat missing values as a category
            self.fairness_df[var] = self.fairness_df[var].fillna('missing')
            self.groups[var]["categories"] = self.fairness_df[var].unique()
            self.groups[var]["indices"] = {}
            for cat in self.groups[var]["categories"]:
                self.groups[var]["indices"][cat] = self.fairness_df[self.fairness_df[var] == cat].index
        print("Groups created")
        return

    def slicer(self, slicing_var: str) -> pd.DataFrame:
        """Method that given a categorical variable, 
            slices the y_true and y_prob into the different categories of the variable"""
        data = {}
        categories = self.groups[slicing_var]["categories"]
        for cat in categories:
            y_true = self.y_true[self.groups[slicing_var][cat]["indices"]]
            y_prob = self.y_prob[self.groups[slicing_var][cat]["indices"]]
            data[cat] = {"y_true": y_true, "y_prob": y_prob}
        return data