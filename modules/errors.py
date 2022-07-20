from collections import defaultdict
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
        self._errors = defaultdict(list)
        self.y_pred = self._calibrate()
        self.combos = self._get_all_combos()

    def _calibrate(self):
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

    def _get_all_combos(self):
        for method, y_pred in self.y_pred.items():
            self.combos[method] = self._all_combos(y_pred)

    def _all_combos(self, pred):
        if re.search("mean.", str(pred.keys())):
                yield ("Calibrated Test Data (Mean)", 
                    pred["mean.Test"], 
                    list(self.test_raw["y"]))
                yield ("Calibrated Test Data (Min)", 
                    pred["min.Test"], 
                    list(self.test_raw["y"]))
                yield ("Calibrated Test Data (Max)", 
                    pred["max.Test"], 
                    list(self.test_raw["y"]))
                yield ("Uncalibrated Test Data", 
                    list(self.test_raw["x"]), 
                    list(self.test_raw["y"]))
                yield ("Calibrated Train Data (Mean)", 
                    pred["mean.Train"], 
                    list(self.test_raw["y"]))
                yield ("Calibrated Train Data (Min)", 
                    pred["min.Train"], 
                    list(self.test_raw["y"]))
                yield ("Calibrated Train Data (Max)", 
                    pred["max.Train"], 
                    list(self.test_raw["y"]))
                yield ("Uncalibrated Train Data", 
                    list(self.test_raw["x"]), 
                    list(self.test_raw["y"]))
                yield ("Calibrated Full Data (Mean)", 
                    pred["mean.Train"] + pred["mean.Test"], 
                    list(self.train_raw["y"]) + list(self.test_raw["y"]))
                yield ("Calibrated Full Data (Min)", 
                    pred["min.Train"] + pred["min.Test"], 
                    list(self.train_raw["y"]) + list(self.test_raw["y"]))
                yield ("Calibrated Full Data (Max)", 
                    pred["max.Train"] + pred["max.Test"], 
                    list(self.train_raw["y"]) + list(self.test_raw["y"]))
                yield ("Uncalibrated Full Data", 
                    list(self.train_raw["x"]) + list(self.test_raw["x"]), 
                    list(self.train_raw["y"]) + list(self.test_raw["y"]))
        else:
            yield ("Calibrated Test Data",
                    pred["Test"],
                    list(self.test_raw["y"]))
            yield ("Calibrated Train Data",
                    pred["Train"],
                    list(self.train_raw["y"]))
            yield ("Calibrated Full Data",
                    pred["Train"] + pred["Test"],
                    list(self.train_raw["y"]) + list(self.test_raw["y"]))
            yield ("Uncalibrated Test Data",
                    list(self.test_raw["x"]),
                    list(self.test_raw["y"]))
            yield ("Uncalibrated Train Data",
                    list(self.train_raw["x"]),
                    list(self.train_raw["y"]))
            yield ("Uncalibrated Full Data",
                    list(self.train_raw["x"]) + list(self.test_raw["x"]),
                    list(self.train_raw["y"]) + list(self.test_raw["y"]))

    def explained_variance_score(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Explained Variance Score")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.explained_variance_score(true, pred)
                        )

    def max(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Max Error")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.max_error(true, pred)
                        )

    def mean_absolute(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Mean Absolute Error")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.mean_absolute_error(true, pred)
                        )

    def root_mean_squared(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Root Mean Squared Error")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.mean_squared_error(true, pred, squared=False)
                        )

    def root_mean_squared_log(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Root Mean Squared Log Error")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.mean_squared_log_error(true, pred, squared=False)
                        )

    def median_absolute(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Median Absolute Error")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.median_absolute_error(true, pred)
                        )

    def mean_absolute_percentage(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Mean Absolute Percentage Error")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.mean_absolute_percentage_error(true, pred)
                        )

    def r2(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("R^2")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.r2_score(true, pred)
                        )

    def mean_poisson_deviance(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Mean Poisson Deviance")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.mean_poisson_deviance(true, pred)
                        )

    def mean_gamma_deviance(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Mean Gamme Deviance")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.mean_gamma_deviance(true, pred)
                        )

    def mean_tweedie_deviance(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Mean Tweedie Deviance")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.mean_tweedie_deviance(true, pred)
                        )

    def mean_pinball_loss(self):
        for method, combo in self.combos.items():
            self._errors[method]["Error"].append("Mean Pinball Loss")
            for name, pred, true in combo:
                self._errors[method][name].append(
                        met.mean_pinball_loss(true, pred)
                        )
    
    def get_errors(self):
        for key, item in self._errors.items():
            self._errors[key] = dict(item)
        return self._errors
