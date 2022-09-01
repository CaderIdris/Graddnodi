import re

import matplotlib.pyplot as plt
import pandas as pd

__author__ = "Idris Hayward"
__copyright__ = "2022, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.1"
__maintainer__ = "Idris Hayward"
__email__ = "CaderIdrisGH@outlook.com"
__status__ = "Indev"

class Figures:
    def __init__(self, train, test, coeffs, errors):
        self.train = train
        self.test = test
        self.coefficients = coeffs
        self.errors = errors
        self.y_pred = self._calibrate()
        print(self.y_pred)
        self.plots = dict()

    def _calibrate(self):
        y_pred_dict = dict()
        column_names = self.coefficients.columns
        for coefficient_set in self.coefficients.iterrows():
            if bool(re.search(r"\'sd\.", str(column_names))):
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
            if re.match(r"coeff\.", key):
                coefficient_keys.append(re.sub(r"coeff\.", "", key))
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
            if re.match(r"coeff\.", key):
                coefficient_keys.append(re.sub(r"coeff\.", "", key))
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

    def linear_reg_plot(self):
        plt.style.use('seaborn-paper')
        fig = plt.figure(figsize=(8,8))
        fig_gs = fig.add_gridspec(
                2, 2, width_ratios=(7,2), height_ratios=(2,7), 
                left=0.1, right=0.9, bottom=0.1, top=0.9,
                wspace=0.05, hspace=0.05
                )

        scatter_ax = fig.add_subplot(fig_gs[1, 0])
        histx_ax = fig.add_subplot(fig_gs[0, 0], sharex=scatter_ax)
        histx_ax.tick_params(axis="x", labelbottom=False)
        histy_ax = fig.add_subplot(fig_gs[1, 1], sharey=scatter_ax)
        histy_ax.tick_params(axis="y", labelleft=False)


