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

import logging

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as ttsplit

logger = logging.getLogger('pymc')
logger.setLevel(logging.ERROR)

class Calibration:
    """ Calibrates one set of measurements against another 

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
        self._coefficients = dict()

    def format_skl(self, mv_keys=list()):
        """ Formats the incoming data for the scikitlearn calibration
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
            combo_string (str): String containing all x variables being
            calibrated
        """
        x_name = self.x_train.columns[1]
        y_name = self.y_train.columns[1]
        combo_string = ["x"]
        x_array = np.array(self.x_train[x_name])[:, np.newaxis]
        for key in mv_keys:
            combo_string.append(f" + {key}")
            secondary = self.x_train[key]
            x_array = np.hstack((x_array, np.array(secondary)[:, np.newaxis]))
        y_array = np.array(self.y_train[y_name])[:, np.newaxis]
        scaler_x = StandardScaler()
        scaler_x.fit(x_array)
        x_array = scaler_x.transform(x_array)
        return x_array, y_array, scaler_x, "".join(combo_string)

    def format_pymc(self, mv_keys):
        x_name = self.x_train.columns[1]
        y_name = self.y_train.columns[1]
        pymc_dataframe = pd.DataFrame()
        pymc_dataframe["x"] = self.x_train[x_name]
        key_string = ["x"]
        bambi_string = ["x"]
        for key in mv_keys:
            key_string.append(f"{key}")
            bambi_string.append(f"{key.replace(' ', '_')}")
            pymc_dataframe[key.replace(' ', '_')] = self.x_train[key]
        pymc_dataframe["y"] = self.y_train[y_name]
        return pymc_dataframe, bambi_string, key_string

    def ols(self, mv_keys=list()):
        """ Performs OLS linear regression on array X against y

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
        slopes_list_scaled, offset_scaled = (
                list(ols_lr.coef_[0]), float(ols_lr.intercept_[0])
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"OLS ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def ridge(self, mv_keys=list()):
        """ Performs ridge linear regression on array X against y

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
        slopes_list_scaled, offset_scaled = (
                list(ridge.coef_[0]), float(ridge.intercept_[0])
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"Ridge ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def lasso(self, mv_keys=list()):
        """ Performs lasso linear regression on array X against y

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
        slopes_list_scaled, offset_scaled = (
                list(lasso.coef_), float(lasso.intercept_)
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"Lasso ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def elastic_net(self, mv_keys=list()):
        """ Performs elastic net linear regression on array X against y

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
        slopes_list_scaled, offset_scaled = (
                list(enet.coef_), float(enet.intercept_)
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"Elastic Net ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def lars(self, mv_keys=list()):
        """ Performs least angle regression on array X against y

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
        slopes_list_scaled, offset_scaled = (
                list(lars_lr.coef_), float(lars_lr.intercept_)
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"Lars ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def lasso_lars(self, mv_keys=list()):
        """ Performs lasso least angle regression on array X against y

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
        lars_lr = lm.LassoLarsCV(normalize=False).fit(x_array, y_array)
        slopes_list_scaled, offset_scaled = (
                list(lars_lr.coef_), float(lars_lr.intercept_)
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"Lasso Lars ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }
    
    def orthogonal_matching_pursuit(self, mv_keys=list()):
        """ Performs orthogonal matching pursuit regression on array X 
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
        omp_lr = lm.OrthogonalMatchingPursuitCV(normalize=False).fit(
                x_array, y_array
                )
        slopes_list_scaled, offset_scaled = (
                list(omp_lr.coef_), float(omp_lr.intercept_)
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"Orthogonal Matching Pursuit ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def ransac(self, mv_keys=list()):
        """ Performs ransac regression on array X against y

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
        slopes_list_scaled, offset_scaled = (
                list(ransac_lr.estimator_.coef_), 
                float(ransac_lr.estimator_.intercept_)
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"RANSAC ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def theil_sen(self, mv_keys=list()):
        """ Performs theil sen regression on array X against y

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
        slopes_list_scaled, offset_scaled = (
                list(theil_sen_lr.coef_), float(theil_sen_lr.intercept_)
                        )
        slopes_list = np.true_divide(
                slopes_list_scaled,
                scaler.scale_
                )
        offset = offset_scaled - np.dot(slopes_list, scaler.mean_)
        slopes_list = list(slopes_list)
        slopes = {"x": slopes_list[0]}
        for index, key in enumerate(mv_keys):
            slopes[key] = slopes_list[index + 1]
        self._coefficients[f"Theil Sen ({combo_string})"] = {
                "Slope": slopes,
                "Offset": offset 
                }

    def maximum_a_posteriori(self):
        """ Performs MAP regression comparing y against x
        """
        pass

    def bayesian(self, mv_keys=list(), family="Gaussian"):
        """ Performs bayesian linear regression (either uni or multivariate)
        on y against x

        Performs bayesian linear regression, both univariate and multivariate,
        on X against y. More details can be found at:
        https://pymc.io/projects/examples/en/latest/generalized_linear_models/
        GLM-robust.html
        """
        # Define model families
        model_families = {
            "Gaussian": 'gaussian',
            "Student T": "t",
            "Bernoulli": 'bernoulli',
            "Beta": "beta",
            "Binomial": "binomial",
            "Gamma": "gamma",
            "Negative Binomial": "negativebinomial",
            "Poisson": "poisson",
            "Inverse Gaussian": "wald"
            }
        pymc_dataframe, bambi_list, combo_list = self.format_pymc(mv_keys)
        # Set priors
        model = bmb.Model(
                formula=f"y ~ {' + '.join(bambi_list)}",
                data=pymc_dataframe,
                family=model_families[family],
                dropna=True,
                )
        fitted = model.fit(
                draws=600,
                tune=600,
                init="adapt_diag",
                progressbar=False
                )
        summary = az.summary(fitted)
        self._coefficients[f"Bayesian ({' + '.join(combo_list)})"] = {
                "Slope": dict(),
                "Offset": dict()
                }
        for combo_key, bambi_key in zip(combo_list, bambi_list):
            self._coefficients[
                    f"Bayesian ({' + '.join(combo_list)})"
                    ]["Slope"][combo_key] = {
                        "Mean": summary.loc[bambi_key, 'mean'],
                        "$\sigma$": summary.loc[bambi_key, 'sd'],
                    }
        self._coefficients[f"Bayesian ({' + '.join(combo_list)})"]["Offset"] = {
                    "Mean": summary.loc['Intercept', 'mean'],
                    "$\sigma$": summary.loc['Intercept', 'sd'],
                }

    def rolling(self):
        """ Performs rolling OLS
        """
        pass

    def appended(self):
        """ Performs appended OLS
        """
        pass


