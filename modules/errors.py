import re

import pandas as pd
from sklearn import metrics as met

class Errors:
    """ Calculates errors between "true" and "predicted" measurements

    Attributes:
        train_raw (DataFrame): Training data

        test_raw (DataFrame): Testing data

        coefficients (DataFrame): Calibration coefficients

        errors (dict): Dictionary of dataframes, each key representing a
        different calibration methods

    Methods:
        calibrate:

        _pymc_calibrate:

        _skl_calibrate:

        explained_variance_score: 

        max:

        mean_absolute:

        root_mean_squared:

        root_mean_squared_log:

        median_absolute:

        mean_absolute_percentage:

        r2:

        mean_poisson_deviance:

        mean_gamma_deviance:

        mean_tweedie_deviance:

        mean_pinball_loss:

    """
    def __init__(self, train, test, coefficients):
        """ Initialise the class

        Keyword Arguments:
            train (DataFrame): Training data

            test (DataFrame): Testing data

            coefficients (DataFrame): Calibration coefficients
        """
        self.train_raw = train
        self.test_raw = test
        self.coefficients = coefficients
        self.errors = dict()
        self.y_pred = self.calibrate()

    def calibrate(self):
        y_pred_dict = dict()
        for coefficient_set in self.coefficients.itertuples():
            if bool(re.search("\'sd\.", str(coefficient_set._fields))):
                y_pred = self._pymc_calibrate(coefficient_set)
            else:
                y_pred = self._skl_calibrate(coefficient_set)
            y_pred_dict[coefficient_set.index] = y_pred
        return y_pred_dict

    def _pymc_calibrate(self, coeffs):
        coefficient_keys_raw = list(coeffs._fields)
        coefficient_keys_raw = [
                element for element in coefficient_keys_raw if element not
                in ["coeff.x", "sd.x", "sd.Intercept", "i.Intercept", "index"]
                ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match("coeff\.", key):
                coefficient_keys.append(re.sub("coeff\.", "", key))
        y_pred_train = self.train["x"] * getattr(coeffs, "coeff.x")
        y_pred_test = self.test["x"] * getattr(coeffs, "coeff.x")
        y_pred = pd.DataFrame(data={
            "mean.Train": y_pred_train.copy(),
            "min.Train": y_pred_train.copy(),
            "max.Train": y_pred_train.copy(),
            "mean.Test": y_pred_test.copy(),
            "min.Test": y_pred_test.copy(),
            "max.Test": y_pred_test.copy(),
            })
        for coeff in coefficient_keys:
            to_add_train = self.train[coeff] * getattr(
                    coeffs, f"coeff.{coeff}"
                    )
            to_add_test = self.test[coeff] * getattr(
                    coeffs, f"coeff.{coeff}"
                    )
            coeff_error_train = self.train[coeff] * (
                    2 * getattr(coeffs, f"sd.{coeff}")
                    )
            coeff_error_test = self.test[coeff] * (
                    2 * getattr(coeffs, f"sd.{coeff}")
                    )
            y_pred["mean.Train"] = y_pred["mean.Train"] + to_add_train
            y_pred["min.Train"] = y_pred["min.Train"] + (
                    to_add_train - coeff_error_train
                    )
            y_pred["max.Train"] = y_pred["max.Train"] + (
                    to_add_train + coeff_error_train
                    )
            y_pred["mean.Test"] = y_pred["mean.Test"] + to_add_test
            y_pred["min.Test"] = y_pred["min.Test"] + (
                    to_add_test - coeff_error_test
                    )
            y_pred["max.Test"] = y_pred["max.Test"] + (
                    to_add_test + coeff_error_test
                    )
        to_add = getattr(coeffs, f"i.Intercept")
        coeff_error = 2 * getattr(coeffs, f"sd.Intercept")

        y_pred["mean.Train"] = y_pred["mean.Train"] + to_add_train
        y_pred["min.Train"] = y_pred["min.Train"] + (
                to_add_train - coeff_error_train
                )
        y_pred["max.Train"] = y_pred["max.Train"] + (
                to_add_train + coeff_error_train
                )
        y_pred["mean.Test"] = y_pred["mean.Test"] + to_add_test
        y_pred["min.Test"] = y_pred["min.Test"] + (
                to_add_test - coeff_error_test
                )
        y_pred["max.Test"] = y_pred["max.Test"] + (
                to_add_test + coeff_error_test
                )
        return y_pred

    def _skl_calibrate(self, coeffs):
        coefficient_keys_raw = list(coeffs._fields)
        coefficient_keys_raw = [
                element for element in coefficient_keys_raw if element not
                in ["coeff.x", "i.Intercept", "index"]
                ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match("coeff\.", key):
                coefficient_keys.append(re.sub("coeff\.", "", key))
        y_pred = pd.DataFrame(data={
            "Train": self.train["x"] * getattr(coeffs, "coeff.x"),
            "Test": self.test["x"] * getattr(coeffs, "coeff.x")
            }
            )
        for coeff in coefficient_keys:
            to_add = self.train[coeff] * getattr(coeffs, f"coeff.{coeff}")
            y_pred["Test"] = y_pred["Test"] + to_add
            y_pred["Train"] = y_pred["Train"] + to_add
        to_add = 2 * getattr(coeffs, "i.Intercept")
        y_pred["Test"] = y_pred["Test"] + to_add
        y_pred["Train"] = y_pred["Train"] + to_add
        return y_pred

    def explained_variance_score(self):
        pass

    def max(self):
        pass

    def mean_absolute(self):
        pass  

    def root_mean_squared(self):
        pass

    def root_mean_squared_log(self):
        pass

    def median_absolute(self):
        pass

    def mean_absolute_percentage(self):
        pass

    def r2(self):
        pass

    def mean_poisson_deviance(self):
        pass

    def mean_gamma_deviance(self):
        pass

    def mean_tweedie_deviance(self):
        pass

    def mean_pinball_loss(self):
        pass
