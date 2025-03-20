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
            y_true = self.y_true[self.groups[slicing_var]["indices"][cat]]
            y_prob = self.y_prob[self.groups[slicing_var]["indices"][cat]]
            data[cat] = {"y_true": y_true, "y_prob": y_prob}
        return data
    

if __name__ == "__main__":
    # Test the class
    y_prob = np.random.rand(1000)
    y_true = np.random.randint(0, 2, 1000)
    race = np.random.choice(['white', 'black', 'asian', 'hispanic'], 1000).reshape(-1,1)
    sex = np.random.choice(["M", "F"], 1000).reshape(-1,1)
    fairness_df = pd.DataFrame(data=np.concatenate((race,sex),axis=1), columns=['race','sex'])

    eq = EquiBoots(y_prob, y_true, fairness_df, fairness_vars=['race','sex'])   

    eq.grouper(groupings_vars=["race","sex"])

    data = eq.slicer("race")

    print(data)