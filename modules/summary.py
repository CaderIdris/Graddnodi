import warnings

import pandas as pd

class Summary:
    def __init__(self, dataframes):
        self._dataframes = dataframes
        if isinstance(self._dataframes, dict):
            dfs = list(self._dataframes.values())
        elif isinstance(self._dataframes, (list, tuple)):
            dfs = self._dataframes
        else:
            raise ValueError(
                    "dataframes arg should be a list or tuple of dataframes" 
                    " or a dictionary with dataframes as values"
                    )
        self.group = pd.concat(dfs).groupby(
                by=self._dataframes[0].index.name,
                level=0
                )

    def mean(self):
        return self.group.mean()

    def median(self):
        return self.group.median()

    def max(self):
        return self.group.max()

    def min(self):
        return self.group.min()

    def _check_valid(self, function_name):
        if isinstance(self._dataframes, dict):
            return True
        else:
            warnings.warn(
                    f"{function_name} expects dataframes to be " 
                    f"represented as a dict"
                    )
            return False

    def diff_from_mean(self):
        if self._check_valid("diff_from_mean"):
            diff_dict = dict()
            mean_df = self.mean()
            for key, df in self._dataframes.items():
                diff_dict[key] = df - mean_df
            return diff_dict
        else:
            return None

    def best_performing(self):
        if self._check_valid("best_performing"):
            count_dict = dict()
            min_df = self.min()
            for key, df in self._dataframes.items():
                count_dict[key] = (df == min_df).values.sum()
            return count_dict
        else:
            return None


