from collections import defaultdict
from pathlib import Path
import re
import sqlite3 as sql

import matplotlib as mpl 
mpl.use("pgf") # Used to make pgf files for latex
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import pandas as pd
from sklearn import metrics as met

class Results:
    """ Calculates errors between "true" and "predicted" measurements, plots
    graphs and returns all results

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
    def __init__(self, train, test, coefficients, comparison_name, x_name=None, y_name=None, x_measurements=None, y_measurements=None):
        """ Initialise the class

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
        y_pred_dict = dict()
        column_names = self.coefficients.columns
        for coefficient_set in self.coefficients.iterrows():
            if bool(re.search("\'sd\.", str(column_names))):
                y_pred = self._pymc_calibrate(coefficient_set[1])
            else:
                y_pred = self._skl_calibrate(coefficient_set[1])
            y_pred_dict[coefficient_set[0]] = y_pred
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

    def _all_combos(self, pred, to_use={"Calibrated Test": True, "Uncalibrated Test": True}):
        combos = list()
        if re.search("mean.", str(pred.keys())):
            if to_use.get("Calibrated Test", False):
                combos.append(("Calibrated Test Data (Mean)", 
                    pred["mean.Test"], 
                    self.test["y"]))
            if to_use.get("Calibrated Test", False) and to_use.get("MinMax", False):
                combos.append(("Calibrated Test Data (Min)", 
                    pred["min.Test"], 
                    self.test["y"]))
                combos.append(("Calibrated Test Data (Max)", 
                    pred["max.Test"], 
                    self.test["y"]))
            if to_use.get("Uncalibrated Test", False):
                combos.append(("Uncalibrated Test Data", 
                    self.test["x"], 
                    self.test["y"]))
            if to_use.get("Calibrated Train", False):
                combos.append(("Calibrated Train Data (Mean)", 
                    pred["mean.Train"], 
                    self.train["y"]))
            if to_use.get("Calibrated Train", False) and to_use.get("MinMax", False):
                combos.append(("Calibrated Train Data (Min)", 
                    pred["min.Train"], 
                    self.train["y"]))
                combos.append(("Calibrated Train Data (Max)", 
                    pred["max.Train"], 
                    self.train["y"]))
            if to_use.get("Uncalibrated Train", False):
                combos.append(("Uncalibrated Train Data", 
                    self.train["x"], 
                    self.train["y"]))
            if to_use.get("Calibrated Full", False):
                combos.append(("Calibrated Full Data (Mean)", 
                    pd.concat([pred["mean.Train"], pred["mean.Test"]]), 
                    pd.concat([self.train["y"], self.test["y"]])))
            if to_use.get("Calibrated Full", False) and to_use.get("MinMax", False):
                combos.append(("Calibrated Full Data (Min)", 
                    pd.concat([pred["min.Train"], pred["min.Test"]]), 
                    pd.concat([self.train["y"], self.test["y"]])))
                combos.append(("Calibrated Full Data (Max)", 
                    pd.concat([pred["max.Train"], pred["max.Test"]]), 
                    pd.concat([self.train["y"], self.test["y"]])))
            if to_use.get("Uncalibrated Full", False):
                combos.append(("Uncalibrated Full Data", 
                    pd.concat([self.train["x"], self.test["x"]]), 
                    pd.concat([self.train["y"], self.test["y"]])))
        else:
            if to_use.get("Calibrated Test", False):
                combos.append(("Calibrated Test Data",
                        pred["Test"],
                        self.test["y"]))
            if to_use.get("Calibrated Train", False):
                combos.append(("Calibrated Train Data",
                        pred["Train"],
                        self.train["y"]))
            if to_use.get("Calibrated Full", False):
                combos.append(("Calibrated Full Data",
                        pd.concat([pred["Train"], pred["Test"]]),
                        pd.concat([self.train["y"],self.test["y"]])))
            if to_use.get("Uncalibrated Test", False):
                combos.append(("Uncalibrated Test Data",
                        self.test["x"],
                        self.test["y"]))
            if to_use.get("Uncalibrated Train", False):
                combos.append(("Uncalibrated Train Data",
                        self.train["x"],
                        self.train["y"]))
            if to_use.get("Uncalibrated Full", False):
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
                plt.style.use('Settings/style.mplstyle')
                fig = plt.figure(figsize=(8,8))
                fig_gs = fig.add_gridspec(
                        2, 2, width_ratios=(7,2), height_ratios=(2,7), 
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.0, hspace=0.0
                        )

                scatter_ax = fig.add_subplot(fig_gs[1, 0])
                histx_ax = fig.add_subplot(fig_gs[0, 0], sharex=scatter_ax)
                histx_ax.axis('off')
                histy_ax = fig.add_subplot(fig_gs[1, 1], sharey=scatter_ax)
                histy_ax.axis('off')

                max_value = max(max(true), max(pred))
                scatter_ax.set_xlim(0, max_value)
                scatter_ax.set_ylim(0, max_value)
                scatter_ax.set_xlabel(f"{self.x_name} ({method})")
                scatter_ax.set_ylabel(f"{self.y_name}")
                scatter_ax.scatter(pred, true)
                number_of_coeffs = np.count_nonzero(~np.isnan(self.coefficients.loc[method].values))
                if bool(re.search("Mean", name)) and not bool(re.search("Uncalibrated", name)) and number_of_coeffs == 4:
                    scatter_ax.axline((0, self.coefficients.loc[method]["i.Intercept"]), slope=self.coefficients.loc[method]["coeff.x"], color='red')
                    scatter_ax.axline((0, self.coefficients.loc[method]["i.Intercept"] + 2*self.coefficients.loc[method]["sd.Intercept"]), slope=(self.coefficients.loc[method]["coeff.x"] + 2*self.coefficients.loc[method]["sd.x"]), color='green')
                    scatter_ax.axline((0, self.coefficients.loc[method]["i.Intercept"] - 2*self.coefficients.loc[method]["sd.Intercept"]), slope=(self.coefficients.loc[method]["coeff.x"] - 2*self.coefficients.loc[method]["sd.x"]), color='green')
                elif bool(re.search("Min", name)) and not bool(re.search("Uncalibrated", name)) and number_of_coeffs == 4:
                    scatter_ax.axline((0, self.coefficients.loc[method]["i.Intercept"] - 2*self.coefficients.loc[method]["sd.Intercept"]), slope=(self.coefficients.loc[method]["coeff.x"] - 2*self.coefficients.loc[method]["sd.x"]), color='red')
                elif bool(re.search("Max", name)) and not bool(re.search("Uncalibrated", name)) and number_of_coeffs == 4:
                    scatter_ax.axline((0, self.coefficients.loc[method]["i.Intercept"] + 2*self.coefficients.loc[method]["sd.Intercept"]), slope=(self.coefficients.loc[method]["coeff.x"] + 2*self.coefficients.loc[method]["sd.x"]), color='red')
                elif not bool(re.search("Uncalibrated", name)) and number_of_coeffs == 2:
                    scatter_ax.axline((0, int(self.coefficients.loc[method]["i.Intercept"])), slope=self.coefficients.loc[method]["coeff.x"], color='red')

                binwidth = 2.5
                xymax = max(np.max(np.abs(pred)), np.max(np.abs(true)))
                lim = (int(xymax/binwidth) + 1) * binwidth

                bins = np.arange(-lim, lim + binwidth, binwidth)
                histx_ax.hist(pred, bins=bins)
                histy_ax.hist(true, bins=bins, orientation='horizontal')
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{name} ({method})")

                self._plots[name][method].append(fig)

    def bland_altman_plot(self, title=None):
        plot_name = "Bland-Altman"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                plt.style.use('Settings/style.mplstyle')
                fig, ax = plt.subplots(figsize=(8,8))
                x_data = np.mean(np.vstack((pred, true)).T, axis=1)
                y_data = np.array(pred) - np.array(true)
                y_mean = np.mean(y_data)
                y_sd = 1.96*np.std(y_data)
                max_diff_from_mean = max((y_data - y_mean).min(), (y_data - y_mean).max(), key=abs)
                ax.set_ylim(y_mean + max_diff_from_mean, y_mean - max_diff_from_mean)
                ax.set_xlabel("Average of Measured and Reference")
                ax.set_ylabel("Difference Between Measured and Reference")
                ax.scatter(x_data, y_data)
                ax.axline((0, y_mean), (1, y_mean), color='red')
                ax.text(max(x_data), y_mean + 1, f"Mean: {y_mean:.2f}", verticalalignment='bottom', horizontalalignment='right')
                ax.axline((0, y_mean + y_sd), (1, y_mean + y_sd), color='blue')
                ax.text(max(x_data), y_mean + y_sd + 1, f"1.96$\\sigma$: {y_mean + y_sd:.2f}", verticalalignment='bottom', horizontalalignment='right')
                ax.axline((0, y_mean - y_sd), (1, y_mean - y_sd), color='blue')
                ax.text(max(x_data), y_mean - y_sd + 1, f"1.96$\\sigma$: -{y_sd:.2f}", verticalalignment='bottom', horizontalalignment='right')
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{name} ({method})")

                self._plots[name][method].append(fig)

    def ecdf_plot(self, title=None): 
        plot_name = "eCDF" 
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                plt.style.use('Settings/style.mplstyle')
                fig, ax = plt.subplots(figsize=(8,8))
                true_x, true_y = ecdf(true)
                pred_x, pred_y = ecdf(pred)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Measurement")
                ax.set_ylabel("Cumulative Total")
                ax.plot(true_x, true_y, linewidth=3, alpha=0.8, label=self.y_name)
                ax.plot(pred_x, pred_y, linestyle='none', marker='.', label=self.x_name)
                ax.legend()
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{name} ({method})")
                self._plots[name][method].append(fig)

    def temp_time_series_plot(self, path, title=None): # This is not a good way to do this
        plt.style.use('Settings/style.mplstyle')
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
                    plot.savefig(f"{directory.as_posix()}/{graph_type}.png")
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
                            name="Errors",
                            con=con,
                            if_exists="replace",
                            index=True
                            )
                    coefficients.to_sql(
                            name="Coefficients",
                            con=con,
                            if_exists="replace",
                            index=True
                            ) 
                    con.close()

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y 
