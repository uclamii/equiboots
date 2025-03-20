import pandas as pd
import numpy as np
from metrics import binary_classification_metrics, multi_class_prevalence, multi_label_classification_metrics, regression_metrics

class EquiBoots:

    def __init__(
        self,
        y_true: np.array,
        y_prob: np.array,
        y_pred: np.array,
        fairness_df: pd.DataFrame,
        fairness_vars: list,
        task: str = "binary_classification",
    ) -> None:

        self.fairness_vars = fairness_vars
        self.task = task
        self.y_true = y_true
        self.y_prob = y_prob
        self.y_pred = y_pred
        self.fairness_df = fairness_df
        self.groups = {}
        self.check_task(task)
        self.check_fairness_vars(fairness_vars)

        pass

    def check_task(self, task):
        if task not in [
            "binary_classification",
            "multi_class_classification",
            "regression",
            "multi_label_classification",
        ]:
            raise ValueError(
                "Invalid task, please supply one of 'binary_classification', 'multi_class_classification', 'regression' or 'multi_label_classification'"
            )

    def check_fairness_vars(self, fairness_vars):
        if fairness_vars is None:
            raise ValueError("fairness_vars cannot be None, please provide a list")
        if not isinstance(fairness_vars, list):
            raise ValueError("fairness_vars must be a list")

    def grouper(self, groupings_vars: list) -> pd.DataFrame:
        """Method that given a list of categorical variables, returns indices of each category."""
        for var in groupings_vars:
            self.groups[var] = {}
            # Replace NaN with 'missing' to treat missing values as a category
            self.fairness_df[var] = self.fairness_df[var].fillna("missing")
            self.groups[var]["categories"] = self.fairness_df[var].unique()
            self.groups[var]["indices"] = {}
            for cat in self.groups[var]["categories"]:
                self.groups[var]["indices"][cat] = self.fairness_df[
                    self.fairness_df[var] == cat
                ].index
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
            y_pred = self.y_pred[self.groups[slicing_var]["indices"][cat]]
            data[cat] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}
        return data


    def get_metrics(self, group_dict):
        

if __name__ == "__main__":
    # Test the class
    y_prob = np.random.rand(1000)
    y_pred = (y_prob > 0.5)
    y_true = np.random.randint(0, 2, 1000)
    race = np.random.choice(["white", "black", "asian", "hispanic"], 1000).reshape(
        -1, 1
    )
    sex = np.random.choice(["M", "F"], 1000).reshape(-1, 1)
    fairness_df = pd.DataFrame(
        data=np.concatenate((race, sex), axis=1), columns=["race", "sex"]
    )

    eq = EquiBoots(y_true, y_prob, y_pred, fairness_df, fairness_vars=["race", "sex"])

    eq.grouper(groupings_vars=["race", "sex"])

    data = eq.slicer("race")

    print(data)
