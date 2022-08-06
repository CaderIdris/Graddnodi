from collections import defaultdict
import re

import pandas as pd
from sklearn import metrics as met

class Errors:
    """ Calculates errors between "true" and "predicted" measurements

    Attributes:
        train (DataFrame): Training data

        test (DataFrame): Testing data

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
        self.train = train
        self.test = test
        self.coefficients = coefficients
        self._errors = defaultdict(lambda: defaultdict(list))
        self.y_pred = self._calibrate()
        self.combos = self._get_all_combos()

    def _calibrate(self):
        y_pred_dict = dict()
        column_names = self.coefficients.columns
        for coefficient_set in self.coefficients.iterrows():
            if bool(re.search("\'sd\.", str(column_names))):
                y_pred = self._pymc_calibrate(coefficient_set[1])
            else:
                y_pred = self._skl_calibrate(coefficient_set[1])
            y_pred_dict[coefficient_set[1]["index"]] = y_pred
        return y_pred_dict

    def _pymc_calibrate(self, coeffs):
        coefficient_keys_raw = list(coeffs.dropna().index)
        coefficient_keys_raw = [
                element for element in coefficient_keys_raw if element not
                in ["coeff.x", "sd.x", "sd.Intercept", "i.Intercept", "index"]
                ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match("coeff\.", key):
                coefficient_keys.append(re.sub("coeff\.", "", key))
        y_pred_train = self.train["x"] * coeffs.get("coeff.x")
        y_pred_test = self.test["x"] * coeffs.get("coeff.x")
        y_pred = {
            "mean.Train": pd.Series(y_pred_train),
            "min.Train": pd.Series(y_pred_train),
            "max.Train": pd.Series(y_pred_train),
            "mean.Test": pd.Series(y_pred_test),
            "min.Test": pd.Series(y_pred_test),
            "max.Test": pd.Series(y_pred_test),
            }
        for coeff in coefficient_keys:
            to_add_train = self.train[coeff] * coeffs.get(f"coeff.{coeff}")
            to_add_test = self.test[coeff] * coeffs.get(f"coeff.{coeff}")
            coeff_error_train = self.train[coeff] * (
                    2 * coeffs.get(f"sd.{coeff}")
                        )
            coeff_error_test = self.test[coeff] * (
                    2 * coeffs.get(f"sd.{coeff}")
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
        to_add_int = coeffs.get(f"i.Intercept")
        int_error = 2 * coeffs.get(f"sd.Intercept")

        y_pred["mean.Train"] = y_pred["mean.Train"] + to_add_int
        y_pred["min.Train"] = y_pred["min.Train"] + (
                to_add_int - int_error
                )
        y_pred["max.Train"] = y_pred["max.Train"] + (
                to_add_int + int_error
                )
        y_pred["mean.Test"] = y_pred["mean.Test"] + to_add_int
        y_pred["min.Test"] = y_pred["min.Test"] + (
                to_add_int - int_error
                )
        y_pred["max.Test"] = y_pred["max.Test"] + (
                to_add_int + int_error
                )
        return y_pred

    def _skl_calibrate(self, coeffs):
        
        coefficient_keys_raw = list(coeffs.dropna().index)
        coefficient_keys_raw = [
                element for element in coefficient_keys_raw if element not
                in ["coeff.x", "i.Intercept", "index"]
                ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match("coeff\.", key):
                coefficient_keys.append(re.sub("coeff\.", "", key))
        y_pred = {
                "Train": pd.Series(self.train["x"]) * coeffs.get("coeff.x"),
                "Test": pd.Series(self.test["x"]) * coeffs.get("coeff.x")
                }
        for coeff in coefficient_keys:
            to_add_test = self.test[coeff] * coeffs.get(f"coeff.{coeff}")
            to_add_train = self.train[coeff] * coeffs.get(f"coeff.{coeff}")
            y_pred["Test"] = y_pred["Test"] + to_add_test
            y_pred["Train"] = y_pred["Train"] + to_add_train
        to_add = coeffs.get("i.Intercept")
        y_pred["Test"] = y_pred["Test"] + to_add
        y_pred["Train"] = y_pred["Train"] + to_add
        return y_pred

    def _get_all_combos(self):
        combos = dict()
        for method, y_pred in self.y_pred.items():
            combos[method] = self._all_combos(y_pred)
        return combos

    def _all_combos(self, pred):
        combos = list()
        if re.search("mean.", str(pred.keys())):
            combos.append(("Calibrated Test Data (Mean)", 
                pred["mean.Test"], 
                self.test["y"]))
            combos.append(("Calibrated Test Data (Min)", 
                pred["min.Test"], 
                self.test["y"]))
            combos.append(("Calibrated Test Data (Max)", 
                pred["max.Test"], 
                self.test["y"]))
            combos.append(("Uncalibrated Test Data", 
                self.test["x"], 
                self.test["y"]))
            combos.append(("Calibrated Train Data (Mean)", 
                pred["mean.Train"], 
                self.train["y"]))
            combos.append(("Calibrated Train Data (Min)", 
                pred["min.Train"], 
                self.train["y"]))
            combos.append(("Calibrated Train Data (Max)", 
                pred["max.Train"], 
                self.train["y"]))
            combos.append(("Uncalibrated Train Data", 
                self.train["x"], 
                self.train["y"]))
            combos.append(("Calibrated Full Data (Mean)", 
                pd.concat([pred["mean.Train"], pred["mean.Test"]]), 
                pd.concat([self.train["y"], self.test["y"]])))
            combos.append(("Calibrated Full Data (Min)", 
                pd.concat([pred["min.Train"], pred["min.Test"]]), 
                pd.concat([self.train["y"], self.test["y"]])))
            combos.append(("Calibrated Full Data (Max)", 
                pd.concat([pred["max.Train"], pred["max.Test"]]), 
                pd.concat([self.train["y"], self.test["y"]])))
            combos.append(("Uncalibrated Full Data", 
                pd.concat([self.train["x"], self.test["x"]]), 
                pd.concat([self.train["y"], self.test["y"]])))
        else:
            combos.append(("Calibrated Test Data",
                    pred["Test"],
                    self.test["y"]))
            combos.append(("Calibrated Train Data",
                    pred["Train"],
                    self.train["y"]))
            combos.append(("Calibrated Full Data",
                    pd.concat([pred["Train"], pred["Test"]]),
                    pd.concat([self.train["y"],self.test["y"]])))
            combos.append(("Uncalibrated Test Data",
                    self.test["x"],
                    self.test["y"]))
            combos.append(("Uncalibrated Train Data",
                    self.train["x"],
                    self.train["y"]))
            combos.append(("Uncalibrated Full Data",
                    pd.concat([self.train["x"], self.test["x"]]),
                    pd.concat([self.train["y"], self.test["y"]])))
        return combos

    def explained_variance_score(self):
        error_name = "Explained Variance Score"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.explained_variance_score(true, pred)
                        )

    def max(self):
        error_name = "Max Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.max_error(true, pred)
                        )

    def mean_absolute(self):
        error_name = "Mean Absolute Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.mean_absolute_error(true, pred)
                        )

    def root_mean_squared(self):
        error_name = "Root Mean Squared Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.mean_squared_error(true, pred, squared=False)
                        )

    def root_mean_squared_log(self):
        error_name = "Root Mean Squared Log Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.mean_squared_log_error(true, pred, squared=False)
                        )

    def median_absolute(self):
        error_name = "Median Absolute Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.median_absolute_error(true, pred)
                        )

    def mean_absolute_percentage(self):
        error_name = "Mean Absolute Percentage Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.mean_absolute_percentage_error(true, pred)
                        )

    def r2(self):
        error_name = "r2"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.r2_score(true, pred)
                        )

    def mean_poisson_deviance(self):
        error_name = "Mean Poisson Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.mean_poisson_deviance(true, pred)
                        )

    def mean_gamma_deviance(self):
        error_name = "Mean Gamma Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.mean_gamma_deviance(true, pred)
                        )

    def mean_tweedie_deviance(self):
        error_name = "Mean Tweedie Deviance"
        for method, combo in self.combos.items():

            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.mean_tweedie_deviance(true, pred)
                        )

    def mean_pinball_loss(self):
        error_name = "Mean Pinball Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                        met.mean_pinball_loss(true, pred)
                        )
    
    def get_errors(self):
        for key, item in self._errors.items():
            self._errors[key] = pd.DataFrame(data=dict(item))
            if "Error" in self._errors[key].columns:
                self._errors[key] = self._errors[key].set_index("Error")
        self._errors = dict(self._errors)
        return self._errors
