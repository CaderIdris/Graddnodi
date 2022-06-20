import datetime as dt
import pandas as pd
import unittest

import modules.idristools as IdrisTools 
import modules.calibration as Calibration

class TestIdrisTools(unittest.TestCase):
    def test_all_combinations(self):
        test_list = ["A", "B", "C", "D", "E", "F", "G"]
        test_combos = IdrisTools.all_combinations(test_list)
        self.assertEqual(len(test_combos), 127)

    # Test all date diff applications
    def test_year_diff(self):
        start = dt.datetime(2015, 1, 1)
        end = dt.datetime(2018, 11, 3)
        test = IdrisTools.DateDifference(start, end)
        self.assertEqual(test.year_difference(), 3)

    def test_month_diff(self):
        start = dt.datetime(2015, 1, 1)
        end = dt.datetime(2018, 11, 3)
        test = IdrisTools.DateDifference(start, end)
        self.assertEqual(test.month_difference(), 46)

    def test_day_diff(self):
        start = dt.datetime(2015, 1, 1)
        end = dt.datetime(2018, 11, 3)
        test = IdrisTools.DateDifference(start, end)
        self.assertEqual(test.day_difference(), 1402)

    def test_add_year(self):
        start = dt.datetime(2015, 1, 1)
        end = dt.datetime(2018, 11, 3)
        test = IdrisTools.DateDifference(start, end)
        self.assertEqual(test.add_year(10), dt.datetime(2025, 1, 1))

    def test_add_month(self):
        start = dt.datetime(2015, 1, 1)
        end = dt.datetime(2018, 11, 3)
        test = IdrisTools.DateDifference(start, end)
        self.assertEqual(test.add_month(1), dt.datetime(2015, 2, 1))

    def test_add_months(self):
        start = dt.datetime(2015, 1, 1)
        end = dt.datetime(2018, 11, 3)
        test = IdrisTools.DateDifference(start, end)
        self.assertEqual(test.add_month(38), dt.datetime(2018, 3, 1))

    def test_add_day(self):
        start = dt.datetime(2015, 1, 1)
        end = dt.datetime(2018, 11, 3)
        test = IdrisTools.DateDifference(start, end)
        self.assertEqual(test.add_days(5), dt.datetime(2015, 1, 6))


def test_data():
    return (
            pd.DataFrame(
                {
                    "Datetime": [0, 1, 2, 3, 4, 5],
                    "X": [0,1,2,3,4,5],
                    "A": [0, 1, 2, 3, 4, 5],
                    "B": [0,1,2,3,4,5],
                    "C": [0,1,2,3,4,5],
                    "D": [0,1,2,3,4,5],
                    "E": [0,1,2,3,4,5],
                    "F": [0,1,2,3,4,5]
                    }
                ),
            pd.DataFrame(
                {
                    "Datetime": [0, 1, 2, 3, 4, 5],
                    "Y": [0,1,2,3,4,5],
                    }
                )
            )

class TestCalibration(unittest.TestCase):
    def test_ols_ulr(self):
        test_data_x, test_data_y = test_data()
        test_ols = Calibration.Calibration(test_data_x, test_data_y)
        test_ols.ols()
        with self.subTest():
            self.assertEqual(round(test_ols.coefficients["OLS (x)"]["Slope"]["x"], 2), 1)
        with self.subTest():
            self.assertEqual(round(test_ols.coefficients["OLS (x)"]["Offset"], 2), 0)

    def test_ols_mlr(self):
        test_data_x, test_data_y = test_data()
        test_ols = Calibration.Calibration(test_data_x, test_data_y)
        test_ols.ols(["A"])
        with self.subTest(): # Test x slope 
            self.assertEqual(round(test_ols.coefficients["OLS (x + A)"]["Slope"]["x"], 2), 0.5)
        with self.subTest(): # Test A slope 
            self.assertEqual(round(test_ols.coefficients["OLS (x + A)"]["Slope"]["A"], 2), 0.5)
        with self.subTest(): # Test intercept
            self.assertEqual(round(test_ols.coefficients["OLS (x + A)"]["Offset"], 2), 0)

    def test_ridge_ulr(self):
        test_data_x, test_data_y = test_data()
        test_ridge = Calibration.Calibration(test_data_x, test_data_y)
        test_ridge.ridge()
        with self.subTest():
            self.assertEqual(round(test_ridge.coefficients["Ridge (x)"]["Slope"]["x"], 2), 1)
        with self.subTest():
            self.assertEqual(round(test_ridge.coefficients["Ridge (x)"]["Offset"], 2), 0)

    def test_ridge_mlr(self):
        test_data_x, test_data_y = test_data()
        test_ridge = Calibration.Calibration(test_data_x, test_data_y)
        test_ridge.ridge(["A"])
        with self.subTest(): # Test x slope 
            self.assertEqual(round(test_ridge.coefficients["Ridge (x + A)"]["Slope"]["x"], 2), 0.5)
        with self.subTest(): # Test A slope 
            self.assertEqual(round(test_ridge.coefficients["Ridge (x + A)"]["Slope"]["A"], 2), 0.5)
        with self.subTest(): # Test intercept
            self.assertEqual(round(test_ridge.coefficients["Ridge (x + A)"]["Offset"], 2), 0)


if __name__ == '__main__':
    unittest.main()
