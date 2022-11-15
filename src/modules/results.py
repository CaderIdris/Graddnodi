from collections import defaultdict
from pathlib import Path
import re
import sqlite3 as sql

import matplotlib as mpl

mpl.use("pgf")  # Used to make pgf files for latex
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.max_open_warning": 0})
import numpy as np
import pandas as pd
from sklearn import metrics as met


class Results:
    """Calculates errors between "true" and "predicted" measurements, plots
    graphs and returns all results

    Attributes:
        train (DataFrame): Training data

        test (DataFrame): Testing data

        coefficients (DataFrame): Calibration coefficients

        _errors (dict): Dictionary of dataframes, each key representing a
        different calibration method

        y_pred (dict): Calibrated x measurements

        combos (list): List of all possible variable and dataset combos

        _plots (dict): All result plots made

        x_name (str): Name of x device

        y_name (str): Name of y device

        x_measurements (pd.DataFrame): x_measurements with associated
        timestamps

        y_measurements (pd.DataFrame): y_measurements with associated
        timestamps

    Methods:
        _calibrate: Calibrate all x measurements with provided coefficients.
        This function splits calibrations depending on whether the coefficients
        were derived using skl or pymc

        _pymc_calibrate: Calibrates x measurements with provided pymc
        coefficients. Returns mean, max and min calibrations.

        _skl_calibrate: Calibrates x measurements with provided skl
        coefficients.

        _get_all_combos: Return all possible combinations of datasets (e.g
        calibrated test, uncalibrated train) for every method

        _all_combos: Return all possible combinations of datasets (e.g
        calibrated test, uncalibrated train) for single method

        explained_variance_score: Calculate the explained variance score
        between the true (y) measurements and all predicted (x) measurements

        max: Calculate the max error between the true (y) measurements and all
        predicted (x) measurements

        mean_absolute: Calculate the mean absolute error between the true (y)
        measurements and all predicted (x) measurements

        root_mean_squared: Calculate the root mean squared error between the
        true (y) measurements and all predicted (x) measurements

        root_mean_squared_log: Calculate the root_mean_squared_log error
        between the true (y) measurements and all predicted (x) measurements

        median_absolute: Calculate the median absolute error between the true
        (y) measurements and all predicted (x) measurements

        mean_absolute_percentage: Calculate the mean absolute percentage error
        between the true (y) measurements and all predicted (x) measurements

        r2: Calculate the r2 score between the true (y) measurements and all
        predicted (x) measurements

        mean_poisson_deviance: Calculate the mean poisson deviance between the
        true (y) measurements and all predicted (x) measurements

        mean_gamma_deviance: Calculate the mean gamma deviance between the true
        (y) measurements and all predicted (x) measurements

        mean_tweedie_deviance: Calculate the mean tweedie deviance between the
        true (y) measurements and all predicted (x) measurements

        mean_pinball_loss: Calculate the mean pinball loss between the true
        (y) measurements and all predicted (x) measurements

        return_errors: Returns dictionary of all recorded errors

        bland_altman_plot: Plots a bland altman graph for all variable
        combinations for all specified datasets using predicted (calibrated x)
        and true (y) data

        linear_reg_plot: Plots a linear regression graph for calibrations that
        only have an x coefficients for all specified datasets using predited
        (calibrated x) and true (y) data

        ecdf_plot: Plots an eCDF graph for all variable combinations for all
        specified dataset using predicted (calibrated x) and true (y) data

        temp_time_series_plot: Temporary way to plot time series, not great

        save_results: Saves errors and coefficients for specific variable and
        dataset to local sqlite3 file

        save_plots: Saves all plots in pgf format
    """

    def __init__(
        self,
        train,
        test,
        coefficients,
        comparison_name,
        x_name=None,
        y_name=None,
        x_measurements=None,
        y_measurements=None,
    ):
        """Initialise the class

        Keyword Arguments:
            train (DataFrame): Training data

            test (DataFrame): Testing data

            coefficients (DataFrame): Calibration coefficients

            comparison_name (String): Name of the comparison
        """
        self.train = train
        self.test = test
        self.coefficients = coefficients
        self._errors = defaultdict(lambda: defaultdict(list))
        self.y_pred = self._calibrate()
        self.combos = self._get_all_combos()
        self._plots = defaultdict(lambda: defaultdict(list))
        self.x_name = x_name
        self.y_name = y_name
        self.x_measurements = x_measurements
        self.y_measurements = y_measurements

    def _calibrate(self):
        """Calibrate all x measurements with provided coefficients.

        All x measurements in both the training and testing dataset need to be
        calibrated with the corresponding coefficients. This function loops
        through all coefficients sets (e.g, calibration using x, calibration
        using x + RH etc) and determines whether the coefficients were obtained
        using scikitlearn or pymc. It then sends them off to the corresponding
        function to be calibrated.
        """
        y_pred_dict = dict()
        column_names = self.coefficients.columns
        for coefficient_set in self.coefficients.iterrows():
            if bool(re.search("'sd\.", str(column_names))):
                y_pred = self._pymc_calibrate(coefficient_set[1])
            else:
                y_pred = self._skl_calibrate(coefficient_set[1])
            y_pred_dict[coefficient_set[0]] = y_pred
        return y_pred_dict

    def _pymc_calibrate(self, coeffs):
        """Calibrates x measurements with provided pymc coefficients. Returns
        mean, max and min calibrations, where max and min are +-2*sd.

        Pymc calibrations don't just provide a coefficient for each variable
        in the form of a mean but also provide a standard deviation on that
        mean. By taking the mean coefficients, mean + 2*sd (max) and mean -
        2*sd (min) we get 3 potential values for the predicted y value.

        Keyword Arguments:
            coeffs (pd.Series): All coefficients to be calibrated with, the
            mean.coeff and sd.coeff correspond to the coefficient mean and
            associated standard deviation. Intercept mean and sd is given with
            i.Intercept and sd.Intercept.
        """
        coefficient_keys_raw = list(coeffs.dropna().index)
        coefficient_keys_raw = [
            element
            for element in coefficient_keys_raw
            if element
            not in ["coeff.x", "sd.x", "sd.Intercept", "i.Intercept", "index"]
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
            coeff_error_train = self.train[coeff] * (2 * coeffs.get(f"sd.{coeff}"))
            coeff_error_test = self.test[coeff] * (2 * coeffs.get(f"sd.{coeff}"))
            y_pred["mean.Train"] = y_pred["mean.Train"] + to_add_train
            y_pred["min.Train"] = y_pred["min.Train"] + (
                to_add_train - coeff_error_train
            )
            y_pred["max.Train"] = y_pred["max.Train"] + (
                to_add_train + coeff_error_train
            )
            y_pred["mean.Test"] = y_pred["mean.Test"] + to_add_test
            y_pred["min.Test"] = y_pred["min.Test"] + (to_add_test - coeff_error_test)
            y_pred["max.Test"] = y_pred["max.Test"] + (to_add_test + coeff_error_test)
        to_add_int = coeffs.get(f"i.Intercept")
        int_error = 2 * coeffs.get(f"sd.Intercept")

        y_pred["mean.Train"] = y_pred["mean.Train"] + to_add_int
        y_pred["min.Train"] = y_pred["min.Train"] + (to_add_int - int_error)
        y_pred["max.Train"] = y_pred["max.Train"] + (to_add_int + int_error)
        y_pred["mean.Test"] = y_pred["mean.Test"] + to_add_int
        y_pred["min.Test"] = y_pred["min.Test"] + (to_add_int - int_error)
        y_pred["max.Test"] = y_pred["max.Test"] + (to_add_int + int_error)
        return y_pred

    def _skl_calibrate(self, coeffs):
        """Calibrate x measurements with provided skl coefficients. Returns
        skl calibration.

        Scikitlearn calibrations provide one coefficient for each variable,
        unlike pymc, so only one predicted signal is returned.

        Keyword Arguments:
            coeffs (pd.Series): All coefficients to be calibrated with, the
            mean.coeff and sd.coeff correspond to the coefficient mean and
            associated standard deviation. Intercept mean and sd is given with
            i.Intercept and sd.Intercept.
        """
        coefficient_keys_raw = list(coeffs.dropna().index)
        coefficient_keys_raw = [
            element
            for element in coefficient_keys_raw
            if element not in ["coeff.x", "i.Intercept", "index"]
        ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match("coeff\.", key):
                coefficient_keys.append(re.sub("coeff\.", "", key))
        y_pred = {
            "Train": pd.Series(self.train["x"]) * coeffs.get("coeff.x"),
            "Test": pd.Series(self.test["x"]) * coeffs.get("coeff.x"),
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
        """Return all possible combinations of datasets.

        This module has the capacity to test the errors of 6 datasets,
        comprised of (un)calibrated test/train/both measurements. There are
        also extra available for pymc calibrations as there is a mean, min and
        max predicted signal. This function generates all the combinations
        and puts them in a dict of lists. This allows for much easier looping
        over the different combinations when calculating errors.

        Keyword Arguments:
            None
        """
        combos = dict()
        for method, y_pred in self.y_pred.items():
            combos[method] = self._all_combos(y_pred)
        return combos

    def _all_combos(
        self, pred, to_use={"Calibrated Test": True, "Uncalibrated Test": True}
    ):
        """Addition to _get_all_combos to get cleaner code

        Keyword arguments:
            pred (dict): Dictionary containing all calibrated signals for a
            single variable combination (e.g x, RH, T)

            to_use (dict): Dict containing all different combos to be used. If
            key is present and corresponding value is True, combo is added to list.
            Keys can be:
                - Calibrated Test: The calibrated test data
                - Uncalibrated Test: The uncalibrated test data
                - Calibrated Train: The calibrated training data
                - Uncalibrated Train: The uncalibrated training data
                - Calibrated Full: The calibrated test + training data
                - Uncalibrated Full: The uncalibrated test + training data
                - MinMax: Use the minimum and maximum values generated by pymc
        """
        combos = list()
        if re.search("mean.", str(pred.keys())):
            if to_use.get("Calibrated Test", False):
                combos.append(
                    ("Calibrated Test Data (Mean)", pred["mean.Test"], self.test["y"])
                )
            if to_use.get("Calibrated Test", False) and to_use.get("MinMax", False):
                combos.append(
                    ("Calibrated Test Data (Min)", pred["min.Test"], self.test["y"])
                )
                combos.append(
                    ("Calibrated Test Data (Max)", pred["max.Test"], self.test["y"])
                )
            if to_use.get("Uncalibrated Test", False):
                combos.append(
                    ("Uncalibrated Test Data", self.test["x"], self.test["y"])
                )
            if to_use.get("Calibrated Train", False):
                combos.append(
                    (
                        "Calibrated Train Data (Mean)",
                        pred["mean.Train"],
                        self.train["y"],
                    )
                )
            if to_use.get("Calibrated Train", False) and to_use.get("MinMax", False):
                combos.append(
                    ("Calibrated Train Data (Min)", pred["min.Train"], self.train["y"])
                )
                combos.append(
                    ("Calibrated Train Data (Max)", pred["max.Train"], self.train["y"])
                )
            if to_use.get("Uncalibrated Train", False):
                combos.append(
                    ("Uncalibrated Train Data", self.train["x"], self.train["y"])
                )
            if to_use.get("Calibrated Full", False):
                combos.append(
                    (
                        "Calibrated Full Data (Mean)",
                        pd.concat([pred["mean.Train"], pred["mean.Test"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
            if to_use.get("Calibrated Full", False) and to_use.get("MinMax", False):
                combos.append(
                    (
                        "Calibrated Full Data (Min)",
                        pd.concat([pred["min.Train"], pred["min.Test"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
                combos.append(
                    (
                        "Calibrated Full Data (Max)",
                        pd.concat([pred["max.Train"], pred["max.Test"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
            if to_use.get("Uncalibrated Full", False):
                combos.append(
                    (
                        "Uncalibrated Full Data",
                        pd.concat([self.train["x"], self.test["x"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
        else:
            if to_use.get("Calibrated Test", False):
                combos.append(("Calibrated Test Data", pred["Test"], self.test["y"]))
            if to_use.get("Calibrated Train", False):
                combos.append(("Calibrated Train Data", pred["Train"], self.train["y"]))
            if to_use.get("Calibrated Full", False):
                combos.append(
                    (
                        "Calibrated Full Data",
                        pd.concat([pred["Train"], pred["Test"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
            if to_use.get("Uncalibrated Test", False):
                combos.append(
                    ("Uncalibrated Test Data", self.test["x"], self.test["y"])
                )
            if to_use.get("Uncalibrated Train", False):
                combos.append(
                    ("Uncalibrated Train Data", self.train["x"], self.train["y"])
                )
            if to_use.get("Uncalibrated Full", False):
                combos.append(
                    (
                        "Uncalibrated Full Data",
                        pd.concat([self.train["x"], self.test["x"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
        return combos

    def explained_variance_score(self):
        """Calculate the explained variance score between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.explained_variance_score
        """
        error_name = "Explained Variance Score"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                    met.explained_variance_score(true, pred)
                )

    def max(self):
        """Calculate the max error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.max_error
        """
        error_name = "Max Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(met.max_error(true, pred))

    def mean_absolute(self):
        """Calculate the mean absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_error
        """
        error_name = "Mean Absolute Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(met.mean_absolute_error(true, pred))

    def root_mean_squared(self):
        """Calculate the root mean squared error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_error
        """
        error_name = "Root Mean Squared Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                    met.mean_squared_error(true, pred, squared=False)
                )

    def root_mean_squared_log(self):
        """Calculate the root mean squared log error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_log_error
        """
        error_name = "Root Mean Squared Log Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                    met.mean_squared_log_error(true, pred, squared=False)
                )

    def median_absolute(self):
        """Calculate the median absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.median_absolute_error
        """
        error_name = "Median Absolute Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(met.median_absolute_error(true, pred))

    def mean_absolute_percentage(self):
        """Calculate the mean absolute percentage error between the true
        values (y) and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_percentage_error
        """
        error_name = "Mean Absolute Percentage Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(
                    met.mean_absolute_percentage_error(true, pred)
                )

    def r2(self):
        """Calculate the r2 between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.r2_score
        """
        error_name = "r2"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(met.r2_score(true, pred))

    def mean_poisson_deviance(self):
        """Calculate the mean poisson deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_poisson_deviance
        """
        error_name = "Mean Poisson Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(met.mean_poisson_deviance(true, pred))

    def mean_gamma_deviance(self):
        """Calculate the mean gamma deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_gamma_deviance
        """
        error_name = "Mean Gamma Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(met.mean_gamma_deviance(true, pred))

    def mean_tweedie_deviance(self):
        """Calculate the mean tweedie deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_tweedie_deviance
        """
        error_name = "Mean Tweedie Deviance"
        for method, combo in self.combos.items():

            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(met.mean_tweedie_deviance(true, pred))

    def mean_pinball_loss(self):
        """Calculate the mean pinball loss between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_pinball_loss
        """
        error_name = "Mean Pinball Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Error"]) == len(self._errors[name][method]):
                    self._errors[name]["Error"].append(error_name)
                self._errors[name][method].append(met.mean_pinball_loss(true, pred))

    def return_errors(self):
        """Returns all calculated errors in dataframe format"""
        for key, item in self._errors.items():
            if not isinstance(self._errors[key], pd.DataFrame):
                self._errors[key] = pd.DataFrame(data=dict(item))
            if "Error" in self._errors[key].columns:
                self._errors[key] = self._errors[key].set_index("Error")
        self._errors = dict(self._errors)
        return self._errors

    def linear_reg_plot(self, title=None):
        plot_name = "Linear Regression"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                if method != "x":
                    self._plots[name][method].append(None)
                    continue
                plt.style.use("Settings/style.mplstyle")
                fig = plt.figure(figsize=(8, 8))
                fig_gs = fig.add_gridspec(
                    2,
                    2,
                    width_ratios=(7, 2),
                    height_ratios=(2, 7),
                    left=0.1,
                    right=0.9,
                    bottom=0.1,
                    top=0.9,
                    wspace=0.0,
                    hspace=0.0,
                )

                scatter_ax = fig.add_subplot(fig_gs[1, 0])
                histx_ax = fig.add_subplot(fig_gs[0, 0], sharex=scatter_ax)
                histx_ax.axis("off")
                histy_ax = fig.add_subplot(fig_gs[1, 1], sharey=scatter_ax)
                histy_ax.axis("off")

                max_value = max(max(true), max(pred))
                scatter_ax.set_xlim(0, max_value)
                scatter_ax.set_ylim(0, max_value)
                scatter_ax.set_xlabel(f"{self.x_name} ({method})")
                scatter_ax.set_ylabel(f"{self.y_name}")
                scatter_ax.scatter(pred, true, color="C0", alpha=0.75)
                number_of_coeffs = np.count_nonzero(
                    ~np.isnan(self.coefficients.loc[method].values)
                )
                if (
                    bool(re.search("Mean", name))
                    and not bool(re.search("Uncalibrated", name))
                    and number_of_coeffs == 4
                ):
                    scatter_ax.axline(
                        (0, self.coefficients.loc[method]["i.Intercept"]),
                        slope=self.coefficients.loc[method]["coeff.x"],
                        color="xkcd:vermillion",
                    )
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[method]["i.Intercept"]
                            + 2 * self.coefficients.loc[method]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[method]["coeff.x"]
                            + 2 * self.coefficients.loc[method]["sd.x"]
                        ),
                        color="xkcd:fresh green",
                    )
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[method]["i.Intercept"]
                            - 2 * self.coefficients.loc[method]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[method]["coeff.x"]
                            - 2 * self.coefficients.loc[method]["sd.x"]
                        ),
                        color="xkcd:fresh green",
                    )
                elif (
                    bool(re.search("Min", name))
                    and not bool(re.search("Uncalibrated", name))
                    and number_of_coeffs == 4
                ):
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[method]["i.Intercept"]
                            - 2 * self.coefficients.loc[method]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[method]["coeff.x"]
                            - 2 * self.coefficients.loc[method]["sd.x"]
                        ),
                        color="xkcd:vermillion",
                    )
                elif (
                    bool(re.search("Max", name))
                    and not bool(re.search("Uncalibrated", name))
                    and number_of_coeffs == 4
                ):
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[method]["i.Intercept"]
                            + 2 * self.coefficients.loc[method]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[method]["coeff.x"]
                            + 2 * self.coefficients.loc[method]["sd.x"]
                        ),
                        color="xkcd:vermillion",
                    )
                elif (
                    not bool(re.search("Uncalibrated", name)) and number_of_coeffs == 2
                ):
                    scatter_ax.axline(
                        (0, int(self.coefficients.loc[method]["i.Intercept"])),
                        slope=self.coefficients.loc[method]["coeff.x"],
                        color="xkcd:vermillion",
                    )

                binwidth = 2.5
                xymax = max(np.max(np.abs(pred)), np.max(np.abs(true)))
                lim = (int(xymax / binwidth) + 1) * binwidth

                bins = np.arange(-lim, lim + binwidth, binwidth)
                histx_ax.hist(pred, bins=bins, color="C0")
                histy_ax.hist(true, bins=bins, orientation="horizontal", color="C0")
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{name} ({method})")

                self._plots[name][method].append(fig)

    def bland_altman_plot(self, title=None):
        plot_name = "Bland-Altman"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                plt.style.use("Settings/style.mplstyle")
                fig, ax = plt.subplots(figsize=(8, 8))
                x_data = np.mean(np.vstack((pred, true)).T, axis=1)
                y_data = np.array(pred) - np.array(true)
                y_mean = np.mean(y_data)
                y_sd = 1.96 * np.std(y_data)
                max_diff_from_mean = max(
                    (y_data - y_mean).min(), (y_data - y_mean).max(), key=abs
                )
                text_adjust = (12 * max_diff_from_mean) / 300
                ax.set_ylim(y_mean - max_diff_from_mean, y_mean + max_diff_from_mean)
                ax.set_xlabel("Average of Measured and Reference")
                ax.set_ylabel("Difference Between Measured and Reference")
                ax.scatter(x_data, y_data, alpha=0.75)
                ax.axline((0, y_mean), (1, y_mean), color="xkcd:vermillion")
                ax.text(
                    max(x_data),
                    y_mean + text_adjust,
                    f"Mean: {y_mean:.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                ax.axline(
                    (0, y_mean + y_sd), (1, y_mean + y_sd), color="xkcd:fresh green"
                )
                ax.text(
                    max(x_data),
                    y_mean + y_sd + text_adjust,
                    f"1.96$\\sigma$: {y_mean + y_sd:.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                ax.axline(
                    (0, y_mean - y_sd), (1, y_mean - y_sd), color="xkcd:fresh green"
                )
                ax.text(
                    max(x_data),
                    y_mean - y_sd + text_adjust,
                    f"1.96$\\sigma$: -{y_sd:.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{name} ({method})")

                self._plots[name][method].append(fig)

    def ecdf_plot(self, title=None):
        plot_name = "eCDF"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                plt.style.use("Settings/style.mplstyle")
                fig, ax = plt.subplots(figsize=(8, 8))
                true_x, true_y = ecdf(true)
                pred_x, pred_y = ecdf(pred)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Measurement")
                ax.set_ylabel("Cumulative Total")
                ax.plot(true_x, true_y, linestyle="none", marker=".", label=self.y_name)
                ax.plot(
                    pred_x,
                    pred_y,
                    linestyle="none",
                    marker=".",
                    alpha=0.8,
                    label=self.x_name,
                )
                ax.legend()
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{name} ({method})")
                self._plots[name][method].append(fig)

    def temp_time_series_plot(
        self, path, title=None
    ):  # This is not a good way to do this
        plt.style.use("Settings/style.mplstyle")
        x_vals = self.x_measurements["Values"]
        y_vals = self.y_measurements["Values"]
        dates = self.x_measurements["Datetime"]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.x_measurements["Datetime"], x_vals, label=self.y_name)
        ax.plot(self.y_measurements["Datetime"], y_vals, label=self.x_name)
        x_null = x_vals.isnull()
        y_null = y_vals.isnull()
        x_or_y_null = np.logical_or(x_null, y_null)
        first_datetime = dates[x_null.loc[~x_or_y_null].index[0]]
        last_datetime = dates[x_null.loc[~x_or_y_null].index[-1]]
        ax.legend()
        ax.set_xlim(first_datetime, last_datetime)
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Concentration")
        fig.savefig(f"{path}/Time Series.pgf")
        fig.savefig(f"{path}/Time Series.png")
        plt.close(fig)

    def save_plots(self, path):
        for key, item in self._plots.items():
            self._plots[key] = pd.DataFrame(data=dict(item))
            if "Plot" in self._plots[key].columns:
                self._plots[key] = self._plots[key].set_index("Plot")
            graph_types = self._plots[key].index.to_numpy()
            for graph_type in graph_types:
                graph_paths = dict()
                for vars, plot in self._plots[key].loc[graph_type].to_dict().items():
                    if plot is None:
                        continue
                    directory = Path(f"{path}/{key}/{vars}")
                    directory.mkdir(parents=True, exist_ok=True)
                    plot.savefig(f"{directory.as_posix()}/{graph_type}.pgf")
                    graph_paths[vars] = f"{directory.as_posix()}/{graph_type}.pgf"
                    plt.close(plot)
                    # key: Data set e.g uncalibrated full data
                    # graph_type: Type of graph e.g Linear Regression
                    # vars: Variables used e.g x + rh
                    # plot: The figure to be saved

    def save_results(self, path):
        for key, item in self._errors.items():
            self._errors[key] = pd.DataFrame(data=dict(item))
            if "Error" in self._errors[key].columns:
                self._errors[key] = self._errors[key].set_index("Error")
                vars_list = self._errors[key].columns.to_list()
                for vars in vars_list:
                    error_results = pd.DataFrame(self._errors[key][vars])
                    coefficients = pd.DataFrame(self.coefficients.loc[vars].T)
                    directory = Path(f"{path}/{key}/{vars}")
                    directory.mkdir(parents=True, exist_ok=True)
                    con = sql.connect(f"{directory.as_posix()}/Results.db")
                    error_results.to_sql(
                        name="Errors", con=con, if_exists="replace", index=True
                    )
                    coefficients.to_sql(
                        name="Coefficients", con=con, if_exists="replace", index=True
                    )
                    con.close()


def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y
