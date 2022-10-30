from collections import defaultdict
import warnings

import pandas as pd

class ArgumentWarning(Warning):
    def __init__(self, message):
        self.message = message 

    def __str__(self):
        return repr(self.message)

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

    def _check_valid(self):
        if isinstance(self._dataframes, dict):
            return True
        else:
            return False

    def diff_from_mean(self):
        if self._check_valid():
            diff_dict = dict()
            mean_df = self.mean()
            for key, df in self._dataframes.items():
                diff_dict[key] = df - mean_df
            return diff_dict
        else:
            warnings.warn("diff_from_mean method requires the _dataframes "
                          "instance to be a dict. Please pass a dict to the "
                          "dataframesm arg when initialising the Summary "
                          "class to use this method", ArgumentWarning)
            return None

    def best_performing(self, summate='key'):
        if self._check_valid():
            count_dict = dict()
            min_df = self.min()
            if summate == 'key':
                for key, df in self._dataframes.items():
                    count_dict[key] = (df == min_df).sum(axis=1).to_dict()
                return count_dict
            elif summate == 'col':
                count_series = pd.Series(dtype=int)
                for key, df in self._dataframes.items():
                    count_series.add((df == min_df).sum(), fill_value=0)
                return count_series.to_dict()
            else:
                warnings.warn("summate keyword argument should be key or col",
                              ArgumentWarning)
        else:
            warnings.warn("best_performing method requires the _dataframes "
                          "instance to be a dict. Please pass a dict to the "
                          "dataframesm arg when initialising the Summary "
                          "class to use this method", ArgumentWarning)
            return None


