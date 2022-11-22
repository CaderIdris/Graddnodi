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


class Errors:
    """Calculates errors between "true" and "predicted" measurements.

    Attributes:
        train (DataFrame): Training data

        test (DataFrame): Testing data

        coefficients (DataFrame): Calibration coefficients

        _errors (dict): Dictionary of dataframes, each key representing a
        different calibration method

        y_pred (dict): Calibrated x measurements

        y_true (dict): Reference measurements

    Methods:
        _calibrate: Calibrate all x measurements with provided coefficients.
        This function splits calibrations depending on whether the coefficients
        were derived using skl or pymc

        _pymc_calibrate: Calibrates x measurements with provided pymc
        coefficients. Returns mean, max and min calibrations.

        _skl_calibrate: Calibrates x measurements with provided skl
        coefficients.

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
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        coefficients: pd.DataFrame,
        to_use: dict = {"Calibrated Test": True, "Uncalibrated Test": True}
    ):
        """Initialise the class

        Keyword Arguments:
            train (DataFrame): Training data

            test (DataFrame): Testing data

            coefficients (DataFrame): Calibration coefficients

            to_use (dict): Datasets to check

        """
        self.train = train
        self.test = test
        self.coefficients = coefficients
        self.y_pred = dict()
        self.y_true = dict()
        self._calibrate(to_use)
        self._errors = self._make_errors_dict()

    def _calibrate(self, to_use):
        """Calibrate all x measurements with provided coefficients.

        All x measurements in both the training and testing dataset need to be
        calibrated with the corresponding coefficients. This function loops
        through all coefficients sets (e.g, calibration using x, calibration
        using x + RH etc) and determines whether the coefficients were obtained
        using scikitlearn or pymc. It then sends them off to the corresponding
        function to be calibrated.
        """
        column_names = self.coefficients.columns
        skl_coeffs = True
        if bool(re.search(r"'sd\.", str(column_names))):
            skl_coeffs = False
        if to_use.get("Calibrated Test"):
            if skl_coeffs:
                self.y_pred["Test (Calibrated)"] = self._skl_calibrate(self.test)
                self.y_true["Test (Calibrated)"] = self.test["y"]
            else:
                self.y_pred["Test (Mean Calibrated)"] = self._pymc_calibrate(self.test)
                self.y_true["Test (Mean Calibrated)"] = self.test["y"]
                if to_use.get("MinMax"):
                    self.y_pred["Test (Max Calibrated)"] = self._pymc_calibrate(self.test, minmax="max")
                    self.y_pred["Test (Min Calibrated)"] = self._pymc_calibrate(self.test, minmax="min")
                    self.y_true["Test (Max Calibrated)"] = self.test["y"] 
                    self.y_true["Test (Min Calibrated)"] = self.test["y"]
        if to_use.get("Calibrated Train"):
            if skl_coeffs:
                self.y_pred["Train (Calibrated)"] = self._skl_calibrate(self.train)
                self.y_true["Train (Calibrated)"] = self.train["y"]
            else:
                self.y_pred["Train (Mean Calibrated)"] = self._pymc_calibrate(self.train)
                self.y_true["Train (Mean Calibrated)"] = self.train["y"]
                if to_use.get("MinMax"):
                    self.y_pred["Train (Max Calibrated)"] = self._pymc_calibrate(self.train, minmax="max")
                    self.y_pred["Train (Min Calibrated)"] = self._pymc_calibrate(self.train, minmax="min")
                    self.y_true["Train (Max Calibrated)"] = self.train["y"] 
                    self.y_true["Train (Min Calibrated)"] = self.train["y"]
        if to_use.get("Calibrated Full"):
            if skl_coeffs:
                self.y_pred["Full (Calibrated)"] = self._skl_calibrate(pd.concat(self.train, self.test))
                self.y_true["Full (Calibrated)"] = pd.concat(self.train, self.test)["y"]
            else:
                self.y_pred["Full (Mean Calibrated)"] = self._pymc_calibrate(pd.concat(self.train, self.test))
                self.y_true["Full (Mean Calibrated)"] = pd.concat(self.train, self.test)["y"]
                if to_use.get("MinMax"):
                    self.y_pred["Full (Max Calibrated)"] = self._pymc_calibrate(pd.concat(self.train, self.test), minmax="max")
                    self.y_pred["Full (Min Calibrated)"] = self._pymc_calibrate(pd.concat(self.train, self.test), minmax="min")
                    self.y_true["Full (Max Calibrated)"] = pd.concat(self.train, self.test)["y"] 
                    self.y_true["Full (Min Calibrated)"] = pd.concat(self.train, self.test)["y"]
        if to_use.get("Uncalibrated Train"):
            self.y_pred["Train (Uncalibrated)"] = {"x": self.train["x"]}
            self.y_true["Train (Uncalibrated)"] = self.train["y"]
        if to_use.get("Uncalibrated Test"):
            self.y_pred["Test (Uncalibrated)"] = {"x": self.test["x"]}
            self.y_true["Test (Uncalibrated)"] = self.test.loc[:, "y"]
        if to_use.get("Uncalibrated Full"):
            self.y_pred["Test (Uncalibrated)"] = {"x": pd.concat(self.test, self.train)["x"]}
            self.y_true["Full (Uncalibrated)"] = pd.concat(self.test, self.train)["y"]

    def _pymc_calibrate(self, data, minmax=None):
        """
        """
        y_pred_dict = dict()
        for index, coeffs in self.coefficients.iterrows():
            # Figure out which coefficients are present, which ones to use and 
            all_keys = list(coeffs.dropna().index)
            coeff_regex = re.compile(r"coeff\.")
            coefficient_keys_raw = list(filter(coeff_regex.match, all_keys))
            coefficient_keys = [re.sub(r"coeff\.", "", label) for label in coefficient_keys_raw]
            coefficient_keys.remove("x") # Remove x and add it on separately
            y_pred: pd.Series = data.loc[:, "x"] * coeffs.get("coeff.x")
            for coeff in coefficient_keys:
                to_add: pd.Series = data.loc[:, coeff] * coeffs.get(f"coeff.{coeff}")
                y_pred = y_pred + to_add
            y_pred = y_pred.add(coeffs.get(f"i.Intercept"))
            if minmax in ["min", "max"]:
                # Just ignore it when minmax isn't called properly.
                # Ideally would warn, TODO?
                mult = 2
                if minmax == "min":
                    mult = -2
                # Quick way to determine +-2*sd
                y_pred = y_pred + (data.loc[:, "x"] * (mult * coeffs.get("sd.x")))
                for coeff in coefficient_keys:
                    to_add: pd.Series = data.loc[:, coeff] * (mult * coeffs.get(f"sd.{coeff}"))
                    y_pred = y_pred + to_add
                y_pred = y_pred + (mult * coeffs.get("sd.Intercept"))
                # Add standard deviations of all coefficients to y_pred to get min or max measurement
            y_pred_dict[index] = y_pred 
        return y_pred_dict

    def _skl_calibrate(self, data):
        """Calibrate x measurements with provided skl coefficients. Returns
        skl calibration.

        Scikitlearn calibrations provide one coefficient for each variable,
        unlike pymc, so only one predicted signal is returned.

        Keyword Arguments:
        """
        y_pred_dict = dict()
        for index, coeffs in self.coefficients.iterrows():
            # Figure out which coefficients are present, which ones to use and 
            all_keys = list(coeffs.dropna().index)
            coeff_regex = re.compile(r"coeff\.")
            coefficient_keys_raw = list(filter(coeff_regex.match, all_keys))
            coefficient_keys = [re.sub(r"coeff\.", "", label) for label in coefficient_keys_raw]
            coefficient_keys.remove("x") # Remove x and add it on separately
            y_pred: pd.Series = data.loc[:, "x"] * coeffs.get("coeff.x")
            for coeff in coefficient_keys:
                to_add: pd.Series = data.loc[:, coeff] * coeffs.get(f"coeff.{coeff}")
                y_pred = y_pred + to_add
            y_pred = y_pred.add(coeffs.get(f"i.Intercept"))
            y_pred_dict[index] = y_pred
        return y_pred_dict

    def _make_errors_dict(self):
        err_dict = dict()
        for dataset in self.y_pred.keys():
            err_dict[dataset] = dict()
        return err_dict

    def explained_variance_score(self):
        """Calculate the explained variance score between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.explained_variance_score
        """
        error_name = "Explained Variance Score"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                try:
                    dset_dict[variables] = met.explained_variance_score(true, pred)
                except TypeError:
                    print(dataset, variables, true, pred)
                    raise Exception
                    
            self._errors[dataset][error_name] = dset_dict

    def max(self):
        """Calculate the max error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.max_error
        """
        error_name = "Max Error"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.max_error(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def mean_absolute(self):
        """Calculate the mean absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_error
        """
        error_name = "Mean Absolute Error"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.mean_absolute_error(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def root_mean_squared(self):
        """Calculate the root mean squared error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_error
        """
        error_name = "Root Mean Squared Error"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.mean_squared_error(true, pred, squared=False)
            self._errors[dataset][error_name] = dset_dict

    def root_mean_squared_log(self):
        """Calculate the root mean squared log error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_log_error
        """
        error_name = "Root Mean Squared Log Error"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.mean_squared_log_error(true, pred, squared=False)
            self._errors[dataset][error_name] = dset_dict

    def median_absolute(self):
        """Calculate the median absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.median_absolute_error
        """
        error_name = "Median Absolute Error"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.median_absolute_error(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def mean_absolute_percentage(self):
        """Calculate the mean absolute percentage error between the true
        values (y) and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_percentage_error
        """
        error_name = "Mean Absolute Percentage Error"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.mean_absolute_percentage_error(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def r2(self):
        """Calculate the r2 between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.r2_score
        """
        error_name = "r2"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.r2_score(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def mean_poisson_deviance(self):
        """Calculate the mean poisson deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_poisson_deviance
        """
        error_name = "Mean Poisson Deviance"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.mean_poisson_deviance(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def mean_gamma_deviance(self):
        """Calculate the mean gamma deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_gamma_deviance
        """
        error_name = "Mean Gamma Deviance"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.mean_gamma_deviance(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def mean_tweedie_deviance(self):
        """Calculate the mean tweedie deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_tweedie_deviance
        """
        error_name = "Mean Tweedie Deviance"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.mean_tweedie_deviance(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def mean_pinball_loss(self):
        """Calculate the mean pinball loss between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_pinball_loss
        """
        error_name = "Mean Pinball Deviance"
        for dataset, predicted in self.y_pred.items():
            dset_dict = dict()
            true = self.y_true.get(dataset)
            for variables, pred in predicted.items():
                dset_dict[variables] = met.mean_pinball_loss(true, pred)
            self._errors[dataset][error_name] = dset_dict

    def return_errors(self):
        """Returns all calculated errors in dataframe format"""
        errors = dict()
        for dataset, dict_of_dicts in self._errors.items():
            dframe = pd.DataFrame.from_dict(dict_of_dicts)
            dframe.index.names = ['Variable']
            errors[dataset] = dframe
        return errors


class Results(Errors):
    """Plots graphs of results from Errors

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

    Methods:
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
    def __init__(self, train, test, coefficients, comparison_name, x_name=None, y_name=None, style="default"):
        Errors.__init__(self, train, test, coefficients)
        self.style = style
        self.x_name = x_name
        self.y_name = y_name

    def linear_reg_plot(self, title=None):
        plot_name = "Linear Regression"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                if method != "x":
                    self._plots[name][method].append(None)
                    continue
                plt.style.use(self.style)
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
                plt.style.use(self.style)
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

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y
