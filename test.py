import unittest
import modules.idristools as IdrisTools 
import modules.calibration as Calibration

class TestIdrisTools(unittest.TestCase):
    def test_test(self):
        self.assertEqual(1+1, 2, "Should be 2")

class TestCalibration(unittest.TestCase):
    def test_ols_lr(self):
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
            self.assertEqual(round(test_ols.coefficients["OLS Univariate Linear Regression"]["Slope"], 2), 1)
        with self.subTest():
            self.assertEqual(round(test_ols.coefficients["OLS Univariate Linear Regression"]["Offset"], 2), 0)

if __name__ == '__main__':
    unittest.main()
