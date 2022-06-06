import unittest
import modules.idristools as IdrisTools 
import modules.calibration as Calibration

class TestIdrisTools(unittest.TestCase):
    def test_all_combinations(self):
        test_list = ["A", "B", "C", "D", "E", "F", "G"]
        test_combos = IdrisTools.all_combinations(test_list)
        self.assertEqual(len(test_combos), 127)

class TestCalibration(unittest.TestCase):
    def test_ols_ulr(self):
        test_data_x = {
            "Measurements": {
                "Values": [0, 1, 2, 3, 4, 5],
                "Timestamps": []
                },
            "Name": "X",
            "Secondary Measurements": {
                "T": {
                    "Values": [0, 1, 2, 3, 4, 5],
                    "Timestamps": [0, 1, 2, 3, 4, 5]
                    }
                }
            }
        test_data_y = {
            "Measurements": {
                "Values": [0, 1, 2, 3, 4, 5],
                "Timestamps": []
                },
            "Name": "Y"
            }
        test_ols = Calibration.Calibration(test_data_x, test_data_y)
        test_ols.ols_linear()
        with self.subTest():
            self.assertEqual(round(test_ols.coefficients["OLS Univariate"]["Slope"], 2), 1)
        with self.subTest():
            self.assertEqual(round(test_ols.coefficients["OLS Univariate"]["Offset"], 2), 0)

    def test_ols_mlr(self):
        test_data_x = {
            "Measurements": {
                "Values": [0, 1, 2, 3, 4, 5],
                "Timestamps": []
                },
            "Name": "X",
            "Secondary Measurements": {
                "A": [0, 1, 2, 3, 4, 5],
                "B": [0,1,2,3,4,5],
                "C": [0,1,2,3,4,5],
                "D": [0,1,2,3,4,5],
                "E": [0,1,2,3,4,5],
                "F": [0,1,2,3,4,5]
                }
            }
        test_data_y = {
            "Measurements": {
                "Values": [0, 1, 2, 3, 4, 5],
                "Timestamps": []
                },
            "Name": "Y"
            }
        test_ols = Calibration.Calibration(test_data_x, test_data_y)
        test_ols.ols_multivariate(["A"])
        with self.subTest(): # Test x slope 
            self.assertEqual(round(test_ols.coefficients["OLS Multivariate (x + A)"]["Slope"][0], 2), 0.5)
        with self.subTest(): # Test A slope 
            self.assertEqual(round(test_ols.coefficients["OLS Multivariate (x + A)"]["Slope"][1], 2), 0.5)
        with self.subTest(): # Test intercept
            self.assertEqual(round(test_ols.coefficients["OLS Multivariate (x + A)"]["Offset"], 2), 0)

if __name__ == '__main__':
    unittest.main()
