"""
"""

__author__ = "Idris Hayward"
__copyright__ = "2022, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.1"
__maintainer__ = "Idris Hayward"
__email__ = "j.d.hayward@surrey.ac.uk"
__status__ = "Indev"

import numpy as np
from sklearn import linear_model as lm
from itertools import combinations

class Calibration:
    """ Calibrates one set of measurements against another 

    Attributes:
        - x (dict): Independent measurements. Keys include:
            - 'Measurements' (dict): Contains two keys:
                - 'Values' (list): Measurements 
                - 'Timestamps' (list): Times measurements made
            - 'Name' (str): Name of device
            - 'Secondary Measurements' (dict): Contains keys representing
            the different secondary variables. Can be empty:
                - *variable* (list): Contains list of measurements
        - y (dict): Dependent measurements. Keys include:
            - 'Measurements' (dict): Contains two keys:
                - 'Values' (list): Measurements 
                - 'Timestamps' (list): Times measurements made
            - 'Name' (str): Name of device

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
    def __init__(self, x_data, y_data):
        """ Initialises the calibration class 

        This class is used to compare one set of measurements against another.
        It also has the capability to perform multivariate calibrations when
        secondary variables are provided.

        Keyword Arguments:
        - x (dict): Independent measurements. Keys include:
            - 'Measurements' (dict): Contains two keys:
                - 'Values' (list): Measurements 
                - 'Timestamps' (list): Times measurements made
            - 'Name' (str): Name of device
            - 'Secondary Measurements' (dict): Contains keys representing
            the different secondary variables. Can be empty:
                - *variable* (list): Contains list of measurements
        - y_data (dict): Dependent measurements. Keys include:
            - 'Measurements' (dict): Contains two keys:
                - 'Values' (list): Measurements 
                - 'Timestamps' (list): Times measurements made
            - 'Name' (str): Name of device
        - coefficients (dict): Results of the calibrations
        """
        self.x = x_data 
        self.y = y_data
        self.coefficients = dict()

    def ols_linear(self):
        """ Performs OLS linear regression comparing y against x
        """
        x_array = np.array(self.x["Measurements"]["Values"])
        y_array = np.array(self.y["Measurements"]["Values"])
        ols_lr = lm.LinearRegression()
        ols_lr.fit(x_array[:, np.newaxis], y_array[:, np.newaxis])
        slope, offset = float(ols_lr.coef_[0]), float(ols_lr.intercept_)
        self.coefficients["OLS Univariate"] = {
                "Slope": slope,
                "Offset": offset 
                }

    def ols_multivariate(self):
        """ Performs multivariate OLS on y against x
        """
        mv_keys = list(self.x["Secondary Measurements"].keys())
        mv_variations = list()
#        for index, key in enumerate(mv_keys[-1]):
#            combo = mv_keys[index:]
#            for size in range(1, len(combo)+1):
#                for begin_index in range(0, len(combo) - size + 1):
#                    print(combo[begin_index:begin_index + size])
        for combo_length in range(1, len(mv_keys) + 1):
            combos = list(combinations(mv_keys, combo_length))
            for combo in combos:
                mv_variations.append(combo)
        print(*mv_variations, sep='\n')
        
        for variation in mv_variations:
            combo_string = "x"
            x_array = np.array(self.x["Measurements"]["Values"])[:, np.newaxis]
            for key in variation:
                combo_string = f"{combo_string} + {key}"
                secondary = self.x["Secondary Measurements"][key]
                x_array = np.hstack((x_array, np.array(secondary)[:, np.newaxis]))
            y_array = np.array(self.y["Measurements"]["Values"])[:, np.newaxis]
            ols_lr = lm.LinearRegression()
            ols_lr.fit(x_array, y_array)
            slope, offset = list(ols_lr.coef_[0]), float(ols_lr.intercept_[0])
            self.coefficients[f"OLS Multivariate ({combo_string})"] = {
                    "Slope": slope,
                    "Offset": offset 
                    }
        print(self.coefficients.keys())

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


