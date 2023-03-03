""" Contains classes and methods used to perform different methods of linear 
regression

This module is used to perform different methods of linear regression on a
dataset (or a training subset), determine all coefficients and then calculate
a range of errors (using the testing subset if available). 

    Classes:
        Calibration: Calibrates one set of measurements against another
"""

__author__ = "Idris Hayward"
__copyright__ = "2022, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.2"
__maintainer__ = "Idris Hayward"
__email__ = "CaderIdrisGH@outlook.com"
__status__ = "Indev"

from collections import defaultdict
import logging

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import re
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as ttsplit

logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)


class Calibration:
    """Calibrates one set of measurements against another

    Attributes:
        - x_train (DataFrame): Independent measurements to train calibration
        on. Columns include:
            "Datetime": Timestamps
            "Values": Main independent measurement
            Remaining columns are secondary measurements which can be used
        - x_test (DataFrame): Independent measurements to test calibration on.
        Columns include:
            "Datetime": Timestamps
            "Values": Main independent measurement
            Remaining columns are secondary measurements which can be used
        - y_train (DataFrame): Dependent measurements to train calibration on.
        Columns include:
            "Datetime": Timestamps
            "Values": Dependent measurements
            Remaining columns are secondary measurements which won't be used
        - y_test (DataFrame): Dependent measurements to test calibration on.
        Columns include:
            "Datetime": Timestamps
            "Values": Dependent measurements
            Remaining columns are secondary measurements which won't be used
        - _coefficients (dict): Results of the calibrations

    Methods:
        - format_skl: Formats data for scikitlearn

        - format_pymc: Formats data for pymc

        - store_coefficients_skl: Stores scikitlearn coefficients

        - store_coefficients_pymc: Stores pymc coefficients

        - ols: Performs OLS linear regression

        - ridge: Performs ridge regression

        - lasso: Performs lasso regression

        - elastic_net: Performs elastic net regression

        - lars: Performs lars regression

        - lasso_lars: Performs lasso lars regression

        - orthogonal_matching_pursuit: Performs OMP regression

        - ransac: Performs ransac regression

        - theil_sen: Performs theil sen regression

        - bayesian: Performs bayesian linear regression

        - rolling: Performs rolling OLS

        - appended: Performs appended OLS

        - return_coefficients: Returns all coefficients as a dict of
        DataFrames

        - return_measurements: Returns all measurements as a dict of
        DataFrames
    """

    def __init__(self, x_data, y_data, split=True, test_size=0.4, seed=72):
        """Initialises the calibration class

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
        print(x_data)
        print(y_data)
        combined_data = x_data.join(
                y_data[["Values"]].rename(
                    columns={"Values": "y"}
                ),
                how="inner",
                lsuffix='x',
                rsuffix='y'
        ).dropna()
        print(combined_data)
        x_data_clean = combined_data.drop(labels=["y"], axis=1)
        y_data_clean = combined_data[["y"]].rename(
            columns={"y": "Values"}
        )
        if split and x_data_clean.shape[0] > 0:
            self.x_train, self.x_test, self.y_train, self.y_test = ttsplit(
                x_data_clean,
                y_data_clean,
                test_size=test_size,
                random_state=seed,
                shuffle=False,
            )
        else:
            self.x_train = x_data_clean
            self.y_train = y_data_clean
            self.x_test = x_data_clean
            self.y_test = y_data_clean
        self._coefficients = defaultdict(pd.DataFrame)

        self.valid_comparison = self.x_train.shape[0] > 0

    def format_skl(self, mv_keys=list()):
        """Formats the incoming data for the scikitlearn calibration
        functions

        The scikitlearn regressors need the input data formatted in a specific
        way. This function also standard scales the x data for better
        performance of some regression techniques.

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used

        Returns:
            tuple representing:
            x_array (np.array): Array representing all independent
            measurements
            y_array (np.array): Array representing all dependent
            measurements
            scaler_x (StandardScaler): StandardScaler object used to transform
            x data
            combo_string (list): List containing all x variables being
            used
        """
        x_name = self.x_train.columns[0]
        y_name = self.y_train.columns[0]
        combo_string = ["x"]
        x_data = {x_name: list(self.x_train[x_name])}
        for key in mv_keys:
            combo_string.append(f"{key}")
            x_data[key] = self.x_train[key]
        y_dataframe = pd.DataFrame(self.y_train[y_name])
        x_dataframe = pd.DataFrame(x_data)
        scaler_x = StandardScaler()
        scaler_x.fit(x_dataframe)
        x_dataframe = scaler_x.transform(x_dataframe)
        return x_dataframe, y_dataframe, scaler_x, combo_string

    def format_pymc(self, mv_keys):
        """Formats the incoming data for the pymc calibration
        functions

        The pymc regressors need the input data formatted in a specific
        way.

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used

        Returns:
            tuple representing:
            pymc_dataframe (DataFrame): All measurements to be used
            bambi_string (list): List containing all x variables being
            used formatted for input in to the model
            combo_string (list): List containing all x variables being
            used
        """
        x_name = self.x_train.columns[0]
        y_name = self.y_train.columns[0]
        pymc_dataframe = pd.DataFrame()
        pymc_dataframe["x"] = self.x_train[x_name]
        key_string = ["x"]
        bambi_string = ["x"]
        for key in mv_keys:
            key_string.append(f"{key}")
            bambi_string.append(re.sub(r"\W*", "o", key))
            pymc_dataframe[re.sub(r"\W*", "o", key)] = self.x_train[key]
        pymc_dataframe["y"] = self.y_train[y_name]
        pymc_dataframe = pymc_dataframe.dropna()
        return pymc_dataframe, bambi_string, key_string

    def store_coefficients_skl(
        self, coeffs_scaled, intercept_scaled, mv_keys, scaler, technique, vars_used
    ):
        """Stores skl coefficients in a DataFrame and stores it in the
        _coefficients attribute

        Keyword arguments:
            coeffs_scaled (list): Coefficients that need to be scaled back
            to original scaling

            intercept_scaled (float): Coefficients that need to be scaled back
            to original scaling

            mv_keys (list): List of multivariate variables used

            scaler (StandardScaler): Contains scaling information to scale
            coefficients to their original values

            technique (stR): Which calibration technique was used?

            vars_used (list): Which variables were used

        """
        coeffs = np.true_divide(coeffs_scaled, scaler.scale_)
        intercept = intercept_scaled - np.dot(coeffs, scaler.mean_)
        results_dict = {"coeff.x": coeffs[0]}
        for index, coeff in enumerate(mv_keys):
            results_dict[f"coeff.{coeff}"] = coeffs[index + 1]
        results_dict["i.Intercept"] = intercept
        results = pd.DataFrame(results_dict, index=[" + ".join(vars_used)])
        self._coefficients[technique] = pd.concat(
            [self._coefficients[technique], results]
        )

    def store_coefficients_pymc(self, summary, bambi_list, combo_list, technique):
        """Stores pymc coefficients in a DataFrame and stores it in the
        _coefficients attribute

        Keyword arguments:
            summary (DataFrame): Output of pymc calibration

            bambi_list (list): All keys representing variables in summary

            combo_list (list): All variable sused in calibration

        """
        results_dict = dict()
        for combo_key, bambi_key in zip(combo_list, bambi_list):
            results_dict[f"coeff.{combo_key}"] = summary.loc[bambi_key, "mean"]
            results_dict[f"sd.{combo_key}"] = summary.loc[bambi_key, "sd"]
        results_dict[f"i.Intercept"] = summary.loc["Intercept", "mean"]
        results_dict[f"sd.Intercept"] = summary.loc["Intercept", "sd"]
        results = pd.DataFrame(results_dict, index=[" + ".join(combo_list)])
        self._coefficients[technique] = pd.concat(
            [self._coefficients[technique], results]
        )

    def ols(self, mv_keys=list()):
        """Performs OLS linear regression on array X against y

        Performs ordinary least squares linear regression, both univariate and
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/
        linear_model.html#ordinary-least-squares

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        ols_lr = lm.LinearRegression()
        ols_lr.fit(x_array, y_array)
        self.store_coefficients_skl(
            ols_lr.coef_[0], ols_lr.intercept_[0], mv_keys, scaler, "OLS", combo_string
        )

    def ridge(self, mv_keys=list()):
        """Performs ridge linear regression on array X against y

        Performs ridge linear regression, both univariate and
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/
        linear_model.html#ridge-regression-and-classification

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        regr_cv = lm.RidgeCV(alphas=np.logspace(-5, 5, 11))
        regr_cv.fit(x_array, y_array)
        ridge_alpha = regr_cv.alpha_
        ridge = lm.Ridge(alpha=ridge_alpha)
        ridge.fit(x_array, y_array)
        self.store_coefficients_skl(
            ridge.coef_[0], ridge.intercept_[0], mv_keys, scaler, "Ridge", combo_string
        )

    def lasso(self, mv_keys=list()):
        """Performs lasso linear regression on array X against y

        Performs lasso linear regression, both univariate and
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/
        linear_model.html#lasso

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        y_array = np.ravel(y_array)
        lasso_cv = lm.LassoCV(alphas=np.logspace(-5, 5, 11))
        lasso_cv.fit(x_array, y_array)
        lasso_alpha = lasso_cv.alpha_
        lasso = lm.Lasso(alpha=lasso_alpha)
        lasso.fit(x_array, y_array)
        self.store_coefficients_skl(
            lasso.coef_, lasso.intercept_, mv_keys, scaler, "LASSO", combo_string
        )

    def elastic_net(self, mv_keys=list()):
        """Performs elastic net linear regression on array X against y

        Performs elastic net linear regression, both univariate and
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/
        linear_model.html#elastic-net

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        y_array = np.ravel(y_array)
        enet_cv = lm.ElasticNetCV(alphas=np.logspace(-5, 5, 11))
        enet_cv.fit(x_array, y_array)
        enet_alpha = enet_cv.alpha_
        enet_l1 = enet_cv.l1_ratio_
        enet = lm.ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1)
        enet.fit(x_array, y_array)
        self.store_coefficients_skl(
            enet.coef_, enet.intercept_, mv_keys, scaler, "Elastic Net", combo_string
        )

    def lars(self, mv_keys=list()):
        """Performs least angle regression on array X against y

        Performs least angle regression, both univariate and
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/
        linear_model.html#least-angle-regression

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        lars_lr = lm.Lars(normalize=False)
        lars_lr.fit(x_array, y_array)
        self.store_coefficients_skl(
            lars_lr.coef_, lars_lr.intercept_, mv_keys, scaler, "LARS", combo_string
        )

    def lasso_lars(self, mv_keys=list()):
        """Performs lasso least angle regression on array X against y

        Performs lasso least angle regression, both univariate and
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/
        linear_model.html#lars-lasso

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        y_array = np.ravel(y_array)
        lasso_lars = lm.LassoLarsCV(normalize=False).fit(x_array, y_array)
        self.store_coefficients_skl(
            lasso_lars.coef_,
            lasso_lars.intercept_,
            mv_keys,
            scaler,
            "LASSO LARS",
            combo_string,
        )

    def orthogonal_matching_pursuit(self, mv_keys=list()):
        """Performs orthogonal matching pursuit regression on array X
        against y

        Performs orthogonal matching pursuit angle regression, only
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/
        linear_model.html#orthogonal-matching-pursuit-omp

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        y_array = np.ravel(y_array)
        omp_lr = lm.OrthogonalMatchingPursuitCV(normalize=False).fit(x_array, y_array)
        self.store_coefficients_skl(
            omp_lr.coef_,
            omp_lr.intercept_,
            mv_keys,
            scaler,
            "Orthogonal Matching Pursuit",
            combo_string,
        )

    def ransac(self, mv_keys=list()):
        """Performs ransac regression on array X against y

        Performs ransac regression, both univariate and
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/
        linear_model.html#ransac-random-sample-consensus

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        y_array = np.ravel(y_array)
        ransac_lr = lm.RANSACRegressor()
        ransac_lr.fit(x_array, y_array)
        self.store_coefficients_skl(
            ransac_lr.estimator_.coef_,
            ransac_lr.estimator_.intercept_,
            mv_keys,
            scaler,
            "RANSAC",
            combo_string,
        )

    def theil_sen(self, mv_keys=list()):
        """Performs theil sen regression on array X against y

        Performs theil sen regression, both univariate and
        multivariate, on X against y. More details can be found at:
        https://scikit-learn.org/stable/modules/linear_model.html
        #theil-sen-estimator-generalized-median-based-estimator

        Coefficients are added to coefficients dict

        Keyword Arguments:
            mv_keys (list): All multivariate variables to be used
        """
        x_array, y_array, scaler, combo_string = self.format_skl(mv_keys)
        y_array = np.ravel(y_array)
        theil_sen_lr = lm.TheilSenRegressor()
        theil_sen_lr.fit(x_array, y_array)
        self.store_coefficients_skl(
            theil_sen_lr.coef_,
            theil_sen_lr.intercept_,
            mv_keys,
            scaler,
            "Theil Sen",
            combo_string,
        )

    def bayesian(self, mv_keys=list(), family="Gaussian"):
        """Performs bayesian linear regression (either uni or multivariate)
        on y against x

        Performs bayesian linear regression, both univariate and multivariate,
        on X against y. More details can be found at:
        https://pymc.io/projects/examples/en/latest/generalized_linear_models/
        GLM-robust.html
        """
        # Define model families
        model_families = {
            "Gaussian": "gaussian",
            "Student T": "t",
            "Bernoulli": "bernoulli",
            "Beta": "beta",
            "Binomial": "binomial",
            "Gamma": "gamma",
            "Negative Binomial": "negativebinomial",
            "Poisson": "poisson",
            "Inverse Gaussian": "wald",
        }
        pymc_dataframe, bambi_list, combo_list = self.format_pymc(mv_keys)
        # Set priors
        model = bmb.Model(
            formula=f"y ~ {' + '.join(bambi_list)}",
            data=pymc_dataframe,
            family=model_families[family],
            dropna=True,
        )
        fitted = model.fit(draws=600, tune=600, init="adapt_diag", progressbar=False)
        summary = az.summary(fitted)
        self.store_coefficients_pymc(
            summary, bambi_list, combo_list, f"Bayesian ({family})"
        )

    def return_coefficients(self):
        """Return all coefficients stored in _coefficients attribute"""
        return dict(self._coefficients)

    def return_measurements(self):
        """Return all test and training measurements"""
        train_measurements = pd.concat(
            [
                self.y_train.rename(columns={"Values": "y"}),
                self.x_train.rename(columns={"Values": "x"}),
            ],
            axis=1,
        )
        test_measurements = pd.concat(
            [
                self.y_test.rename(columns={"Values": "y"}),
                self.x_test.rename(columns={"Values": "x"}),
            ],
            axis=1,
        )
        return {"Train": train_measurements, "Test": test_measurements}

    def rolling(self):
        """Performs rolling OLS"""
        pass

    def appended(self):
        """Performs appended OLS"""
        pass
