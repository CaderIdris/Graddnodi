import re

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
        for coefficient_set in self.coefficients.itertuples():
            if bool(re.search("\'sd\.", str(coefficient_set._fields))):
                y_pred = self._pymc_calibrate(coefficient_set)
            else:
                y_pred = self._skl_calibrate(coefficient_set)

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
