"""
"""

__author__ = "Idris Hayward"
__copyright__ = "2022, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.1"
__maintainer__ = "Idris Hayward"
__email__ = "j.d.hayward@surrey.ac.uk"
__status__ = "Indev"

class Calibration:
    """
    """
    def __init__(self, x_data, y_data, y_mv_data=[]):
        """ Initialises the calibration class 

        This class is used to compare one set of measurements against another.
        It also has the capability to perform multivariate calibrations when
        secondary variables are provided.

        """
        pass

    def ols_linear(self):
        """ Performs OLS linear regression comparing y against x
        """
        pass

    def maximum_a_posteriori(self):
        """ Performs MAP regression comparing y against x
        """
        pass

    def bayesian(self):
        """ Performs bayesian linear regression (either uni or multivariate)
        on y against x
        """
        pass

    def robust_bayesian(self):
        """ Performs robust bayesian linear regression (either uni or multi)
        on y against x
        """
        pass

    def multivariate(self):
        """ Performs multivariate OLS on y against x
        """
        pass

    def rolling(self):
        """ Performs rolling OLS
        """
        pass

    def appended(self):
        """ Performs appended OLS
        """
        pass
