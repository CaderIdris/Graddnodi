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
        explained_variance_score: 

        max:

        mean_absolute:

        mean_squared:

        mean_squared_log:

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
        self.errors = dict()
