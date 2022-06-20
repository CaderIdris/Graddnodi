"""
"""

__author__ = "Idris Hayward"
__copyright__ = "2022, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.1"
__maintainer__ = "Idris Hayward"
__email__ = "CaderIdrisGH@outlook.com"
__status__ = "Indev"

import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as ttsplit


class Calibration:
    """ Calibrates one set of measurements against another 

    Attributes:
        - x_train (DataFrame): Independent measurements to train calibration
        on. Columns include:
            TBA
        - x_test (DataFrame): Independent measurements to test calibration on.
        Columns include:
            TBA
        - y_train (DataFrame): Dependent measurements to train calibration on.
        Columns include:
            TBA
        - y_test (DataFrame): Dependent measurements to test calibration on.
        Columns include:
            TBA
        - coefficients (dict): Results of the calibrations

    Methods:
        - ols_linear: Performs OLS linear regression 

        - maximum_a_posteriori: Performs MAP regression 

        - bayesian: Performs bayesian linear regression (uni or multi)

        - robust_bayesian: Performs robust bayesian linear regression (uni or
        multi)

        - multivariate: Performs multivariate OLS linear regression

        - rolling: Performs rolling OLS 

        - appended: Performs appended OLS
    """
    def __init__(self, x_data, y_data, split=True, test_size=0.4, seed=72):
        """ Initialises the calibration class 

        This class is used to compare one set of measurements against another.
        It also has the capability to perform multivariate calibrations when
        secondary variables are provided.

        Keyword Arguments:
        - x_data (DataFrame): Independent measurements. Column include:
            TBA
        - y_data (DataFrame): Dependent measurements. Keys include:
            TBA
        - split(bool): Split the dataset? Default: True
        - test_size (float): Proportion of the data to use for testing. Use value 
        greater than 0 but less than 1. Defaults to 0.4
        - seed (int): Seed to use when deciding how to split variables,
        ensures consistency between runs. Defaults to 72.
        """
        if split:
            self.x_train, self.x_test, self.y_train, self.y_test = ttsplit(
                x_data, y_data, test_size=test_size, random_state=seed
                    )
        else:
            self.x_train = x_data
            self.y_train = y_data
            self.x_test = x_data
            self.y_test = y_data
        self.coefficients = dict()

    def ols(self, mv_keys=list()):
        """ Performs OLS linear regression on array X against y
        """
        x_name = self.x_train.columns[1]
        y_name = self.y_train.columns[1]
        mv_variations = list()
        combo_string = "x"
        x_array = np.array(self.x_train[x_name])[:, np.newaxis]
        for key in mv_keys:
            combo_string = f"{combo_string} + {key}"
            secondary = self.x_train[key]
            x_array = np.hstack((x_array, np.array(secondary)[:, np.newaxis]))
        y_array = np.array(self.y_train[y_name])[:, np.newaxis]
        ols_lr = lm.LinearRegression()
        ols_lr.fit(x_array, y_array)
        slopes_list, offset = list(ols_lr.coef_[0]), float(ols_lr.intercept_[0])
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self.coefficients[f"OLS ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def ridge(self, mv_keys=list()):
        """ Performs ridge linear regression on array X against y
        """
        x_name = self.x_train.columns[1]
        y_name = self.y_train.columns[1]
        mv_variations = list()
        combo_string = "x"
        x_array = np.array(self.x_train[x_name])[:, np.newaxis]
        for key in mv_keys:
            combo_string = f"{combo_string} + {key}"
            secondary = self.x_train[key]
            x_array = np.hstack((x_array, np.array(secondary)[:, np.newaxis]))
        y_array = np.array(self.y_train[y_name])[:, np.newaxis]
        regr_cv = lm.RidgeCV(alphas=np.logspace(-5, 5, 11))
        regr_cv.fit(x_array, y_array)
        ridge_alpha = regr_cv.alpha_
        ridge = lm.Ridge(alpha=ridge_alpha)
        ridge.fit(x_array, y_array)
        slopes_list, offset = list(ridge.coef_[0]), float(ridge.intercept_[0])
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self.coefficients[f"Ridge ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    # Lasso?
    # Elastic-net?
    # Multi task elastic net?
    # Random sample consensus?
    # Theil sen?


    def maximum_a_posteriori(self):
        """ Performs MAP regression comparing y against x
        """
        pass

    def bayesian(self):
        """ Performs bayesian linear regression (either uni or multivariate)
        on y against x
        """
        pass

    def robust_bayesian(self):
        """ Performs robust bayesian linear regression (either uni or multi)
        on y against x
        """
        pass


    def rolling(self):
        """ Performs rolling OLS
        """
        pass

    def appended(self):
        """ Performs appended OLS
        """
        pass


