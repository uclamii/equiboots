import pandas as pd
import numpy as np

class EquiBoots:
    
    def __init__(self, y_prob: np.array, y_true: np.array, fairness_df: pd.DataFrame, task: str ='binary_classification', fariness_vars: list=[None]) -> None:
        
        pass