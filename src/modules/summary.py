""" Takes a collection of DataFrames and performs summary statistics on them.

Classes:
    ArgumentWarning:
        Inherits from warning. Used to warn when methods used on dictionary of
        DataFrames called when list/tuple of dataframes given.

    Summary:
        Performs summary statistics on provided dataframes

"""
import warnings

import matplotlib.pyplot as plt
import pandas as pd


class ArgumentWarning(Warning):
    """ Used to warn when methods requiring specific argument type receive
    another
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class Summary:
    """ Generates summary statistics for provided dataframes

    Attributes:
        - _dataframes (list,tuple,dict): Dataframes to perform summary stats on

        - group (pd.DataFrame.group): 

    """
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
        self.group = pd.concat(dfs).groupby(by=dfs[0].index.name, level=0)

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
            warnings.warn(
                "diff_from_mean method requires the _dataframes "
                "instance to be a dict. Please pass a dict to the "
                "dataframesm arg when initialising the Summary "
                "class to use this method",
                ArgumentWarning,
            )
            return None

    def best_performing(self, summate="key"):
        if self._check_valid():
            count_dict = dict()
            min_df = self.min()
            if summate == "all":
                for key, df in self._dataframes.items():
                    count_dict_temp = (df.eq(min_df)).sum(axis=1).to_dict()
                    keys = list(count_dict_temp.keys())
                    keys.sort(key=lambda a: a.count("+"))
                    count_dict[key] = {key: count_dict_temp[key] for key in keys}
                return count_dict
            elif summate == "key":
                for key, df in self._dataframes.items():
                    count_dict[key] = (df.eq(min_df)).sum().sum()
                keys = list(count_dict.keys())
                keys.sort(key=lambda a: a.count("+"))
                return {key: count_dict[key] for key in keys}
            elif summate == "row":
                for key, df in self._dataframes.items():
                    column_eq_min = df.eq(min_df).sum(axis=1)
                    if column_eq_min.max() != 0:
                        best_columns = column_eq_min[
                            column_eq_min == column_eq_min.max()
                        ]
                        for col in best_columns.index:
                            if count_dict.get(col) is None:
                                count_dict[col] = 1
                            else:
                                count_dict[col] = count_dict[col] + 1
                keys = list(count_dict.keys())
                keys.sort(key=lambda a: a.count("+"))
                return {key: count_dict[key] for key in keys}
            else:
                warnings.warn(
                    "summate keyword argument should be key or col", ArgumentWarning
                )
        else:
            warnings.warn(
                "best_performing method requires the _dataframes "
                "instance to be a dict. Please pass a dict to the "
                "dataframesm arg when initialising the Summary "
                "class to use this method",
                ArgumentWarning,
            )
            return None

