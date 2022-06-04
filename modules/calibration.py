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
    """ Calibrates one set of measurements against another 

    Attributes:
        - x (dict): Measurements to be calibrated against. Keys include:
            - 'Measurements' (dict): Contains two keys:
                - 'Values' (list): Measurements 
                - 'Timestamps' (list): Times measurements made
            - 'Name' (str): Name of device
        - y (dict): Measurements to be calibrated. Keys include:
            - 'Measurements' (dict): Contains two keys:
                - 'Values' (list): Measurements 
                - 'Timestamps' (list): Times measurements made
            - 'Name' (str): Name of device
            - 'Secondary Measurements' (dict): Contains keys representing
            the different secondary variables. Can be empty:
                - *variable* (list): Contains list of measurements

    Methods:
        - ols_linear: Performs OLS linear regression 

        - maximum_a_posteriori: Performs MAP regression 

        - bayesian: Performs bayesian linear regression (uni or multi)

        - robust_bayesian: Performs robust bayesian linear regression (uni or
        multi)

        - multivariate: Performs multivariate OLS linear regression

        - rolling: Performs rolling OLS 

        - appended: Performs appended OLS
    """
    def __init__(self, x_data, y_data):
        """ Initialises the calibration class 

        This class is used to compare one set of measurements against another.
        It also has the capability to perform multivariate calibrations when
        secondary variables are provided.

        Keyword Arguments:
            - x_data (dict): Dict of the data that y is to be calibrated
            against. Keys includes:
                - 'Measurements' (dict): Contains two keys:
                    - 'Values' (list): Measurements 
                    - 'Timestamps' (list): Times measurements made
                - 'Name' (str): Name of device
    
            - y_data (dict): Dict of the data that is to be calibrated
            Keys includes:
                - 'Measurements' (dict): Contains two keys:
                    - 'Values' (list): Measurements 
                    - 'Timestamps' (list): Times measurements made
                - 'Name' (str): Name of device
                - 'Secondary Measurements' (dict): Contains keys representing
                the different secondary variables. Can be empty:
                    - *variable* (list): Contains list of measurements
                     

        """
        self.x = x_data 
        self.y = y_data 

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
