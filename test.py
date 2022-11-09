import datetime as dt
import numpy as np
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
    data = np.array(range(0, 100000)) / 100
    return (
        pd.DataFrame(
            {
                "Datetime": data.copy(),
                "Values": data.copy(),
                "A": data.copy(),
                "B": data.copy(),
                "C": data.copy(),
                "D": data.copy(),
                "E": data.copy(),
                "F": data.copy(),
            }
        ),
        pd.DataFrame(
            {
                "Datetime": data.copy(),
                "Values": data.copy(),
            }
        ),
    )


class TestCalibration(unittest.TestCase):
    def test_ols(self):

        test_data_x, test_data_y = test_data()
        test_ols = Calibration.Calibration(test_data_x, test_data_y)
        test_ols.ols()
        with self.subTest():
            self.assertEqual(
                round(test_ols._coefficients["OLS"].loc["x", "coeff.x"], 2), 1
            )
        with self.subTest():
            self.assertEqual(
                round(test_ols._coefficients["OLS"].loc["x", "i.Intercept"], 2), 0
            )

    def test_ols_mlr(self):

        test_data_x, test_data_y = test_data()
        test_ols = Calibration.Calibration(test_data_x, test_data_y)
        test_ols.ols(["A"])
        with self.subTest():
            self.assertEqual(
                round(test_ols._coefficients["OLS"].loc["x + A", "coeff.x"], 2), 0.5
            )
        with self.subTest():
            self.assertEqual(
                round(test_ols._coefficients["OLS"].loc["x + A", "coeff.A"], 2), 0.5
            )
        with self.subTest():
            self.assertEqual(
                round(test_ols._coefficients["OLS"].loc["x + A", "i.Intercept"], 2), 0
            )

    def test_ridge_ulr(self):

        test_data_x, test_data_y = test_data()
        test_ridge = Calibration.Calibration(test_data_x, test_data_y)
        test_ridge.ridge()
        with self.subTest():
            self.assertEqual(
                round(test_ridge._coefficients["Ridge"].loc["x", "coeff.x"], 2), 1
            )
        with self.subTest():
            self.assertEqual(
                round(test_ridge._coefficients["Ridge"].loc["x", "i.Intercept"], 2), 0
            )

    def test_ridge_mlr(self):

        test_data_x, test_data_y = test_data()
        test_ridge = Calibration.Calibration(test_data_x, test_data_y)
        test_ridge.ridge(["A"])
        with self.subTest():
            self.assertEqual(
                round(test_ridge._coefficients["Ridge"].loc["x + A", "coeff.x"], 2), 0.5
            )
        with self.subTest():
            self.assertEqual(
                round(test_ridge._coefficients["Ridge"].loc["x + A", "coeff.A"], 2), 0.5
            )
        with self.subTest():
            self.assertEqual(
                round(test_ridge._coefficients["Ridge"].loc["x + A", "i.Intercept"], 2),
                0,
            )

    def test_lasso_ulr(self):

        test_data_x, test_data_y = test_data()
        test_lasso = Calibration.Calibration(test_data_x, test_data_y)
        test_lasso.lasso()
        with self.subTest():
            self.assertEqual(
                round(test_lasso._coefficients["LASSO"].loc["x", "coeff.x"], 2), 1
            )
        with self.subTest():
            self.assertEqual(
                round(test_lasso._coefficients["LASSO"].loc["x", "i.Intercept"], 2), 0
            )

    def test_lasso_mlr(self):

        test_data_x, test_data_y = test_data()
        test_lasso = Calibration.Calibration(test_data_x, test_data_y)
        test_lasso.lasso(["A"])
        with self.subTest():
            self.assertEqual(
                round(test_lasso._coefficients["LASSO"].loc["x + A", "coeff.x"], 2), 1
            )
        with self.subTest():
            self.assertEqual(
                round(test_lasso._coefficients["LASSO"].loc["x + A", "coeff.A"], 2), 0
            )
        with self.subTest():
            self.assertEqual(
                round(test_lasso._coefficients["LASSO"].loc["x + A", "i.Intercept"], 2),
                0,
            )

    def test_elastic_net_ulr(self):

        test_data_x, test_data_y = test_data()
        test_elastic_net = Calibration.Calibration(test_data_x, test_data_y)
        test_elastic_net.elastic_net()
        with self.subTest():
            self.assertEqual(
                round(
                    test_elastic_net._coefficients["Elastic Net"].loc["x", "coeff.x"], 2
                ),
                1,
            )
        with self.subTest():
            self.assertTrue(
                round(
                    test_elastic_net._coefficients["Elastic Net"].loc[
                        "x", "i.Intercept"
                    ],
                    2,
                )
                <= 0.1
            )

    def test_elastic_net_mlr(self):

        test_data_x, test_data_y = test_data()
        test_elastic_net = Calibration.Calibration(test_data_x, test_data_y)
        test_elastic_net.elastic_net(["A"])
        with self.subTest():
            self.assertEqual(
                round(
                    test_elastic_net._coefficients["Elastic Net"].loc[
                        "x + A", "coeff.x"
                    ],
                    2,
                ),
                1,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_elastic_net._coefficients["Elastic Net"].loc[
                        "x + A", "coeff.A"
                    ],
                    2,
                ),
                0,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_elastic_net._coefficients["Elastic Net"].loc[
                        "x + A", "i.Intercept"
                    ],
                    2,
                ),
                0,
            )

    def test_lars_ulr(self):

        test_data_x, test_data_y = test_data()
        test_lars = Calibration.Calibration(test_data_x, test_data_y)
        test_lars.lars()
        with self.subTest():
            self.assertEqual(
                round(test_lars._coefficients["LARS"].loc["x", "coeff.x"], 2), 1
            )
        with self.subTest():
            self.assertEqual(
                round(test_lars._coefficients["LARS"].loc["x", "i.Intercept"], 2), 0
            )

    def test_lars_mlr(self):

        test_data_x, test_data_y = test_data()
        test_lars = Calibration.Calibration(test_data_x, test_data_y)
        test_lars.lars(["A"])
        with self.subTest():
            self.assertEqual(
                round(test_lars._coefficients["LARS"].loc["x + A", "coeff.x"], 2), 1
            )
        with self.subTest():
            self.assertEqual(
                round(test_lars._coefficients["LARS"].loc["x + A", "coeff.A"], 2), 0
            )
        with self.subTest():
            self.assertEqual(
                round(test_lars._coefficients["LARS"].loc["x + A", "i.Intercept"], 2), 0
            )

    def test_lasso_lars_ulr(self):

        test_data_x, test_data_y = test_data()
        test_lasso_lars = Calibration.Calibration(test_data_x, test_data_y)
        test_lasso_lars.lasso_lars()
        with self.subTest():
            self.assertEqual(
                round(
                    test_lasso_lars._coefficients["LASSO LARS"].loc["x", "coeff.x"], 2
                ),
                1,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_lasso_lars._coefficients["LASSO LARS"].loc["x", "i.Intercept"],
                    2,
                ),
                0,
            )

    def test_lasso_lars_mlr(self):

        test_data_x, test_data_y = test_data()
        test_lasso_lars = Calibration.Calibration(test_data_x, test_data_y)
        test_lasso_lars.lasso_lars(["A"])
        with self.subTest():
            self.assertEqual(
                round(
                    test_lasso_lars._coefficients["LASSO LARS"].loc["x + A", "coeff.x"],
                    2,
                ),
                1,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_lasso_lars._coefficients["LASSO LARS"].loc["x + A", "coeff.A"],
                    2,
                ),
                0,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_lasso_lars._coefficients["LASSO LARS"].loc[
                        "x + A", "i.Intercept"
                    ],
                    2,
                ),
                0,
            )

    # ULR not possible
    def test_orthogonal_matching_pursuit_mlr(self):

        test_data_x, test_data_y = test_data()
        test_orthogonal_matching_pursuit = Calibration.Calibration(
            test_data_x, test_data_y
        )
        test_orthogonal_matching_pursuit.orthogonal_matching_pursuit(["A"])
        with self.subTest():
            self.assertEqual(
                round(
                    test_orthogonal_matching_pursuit._coefficients[
                        "Orthogonal Matching Pursuit"
                    ].loc["x + A", "coeff.x"],
                    2,
                ),
                1,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_orthogonal_matching_pursuit._coefficients[
                        "Orthogonal Matching Pursuit"
                    ].loc["x + A", "coeff.A"],
                    2,
                ),
                0,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_orthogonal_matching_pursuit._coefficients[
                        "Orthogonal Matching Pursuit"
                    ].loc["x + A", "i.Intercept"],
                    2,
                ),
                0,
            )

    def test_ransac_ulr(self):

        test_data_x, test_data_y = test_data()
        test_ransac = Calibration.Calibration(test_data_x, test_data_y)
        test_ransac.ransac()
        with self.subTest():
            self.assertEqual(
                round(test_ransac._coefficients["RANSAC"].loc["x", "coeff.x"], 2), 1
            )
        with self.subTest():
            self.assertEqual(
                round(test_ransac._coefficients["RANSAC"].loc["x", "i.Intercept"], 2), 0
            )

    def test_ransac_mlr(self):

        test_data_x, test_data_y = test_data()
        test_ransac = Calibration.Calibration(test_data_x, test_data_y)
        test_ransac.ransac(["A"])
        with self.subTest():
            self.assertEqual(
                round(test_ransac._coefficients["RANSAC"].loc["x + A", "coeff.x"], 2),
                0.5,
            )
        with self.subTest():
            self.assertEqual(
                round(test_ransac._coefficients["RANSAC"].loc["x + A", "coeff.A"], 2),
                0.5,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_ransac._coefficients["RANSAC"].loc["x + A", "i.Intercept"], 2
                ),
                0,
            )

    def test_theil_sen_ulr(self):

        test_data_x, test_data_y = test_data()
        test_theil_sen = Calibration.Calibration(test_data_x, test_data_y)
        test_theil_sen.theil_sen()
        with self.subTest():
            self.assertEqual(
                round(test_theil_sen._coefficients["Theil Sen"].loc["x", "coeff.x"], 2),
                1,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_theil_sen._coefficients["Theil Sen"].loc["x", "i.Intercept"], 2
                ),
                0,
            )

    def test_theil_sen_mlr(self):

        test_data_x, test_data_y = test_data()
        test_theil_sen = Calibration.Calibration(test_data_x, test_data_y)
        test_theil_sen.theil_sen(["A"])
        with self.subTest():
            self.assertEqual(
                round(
                    test_theil_sen._coefficients["Theil Sen"].loc["x + A", "coeff.x"], 2
                ),
                0.5,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_theil_sen._coefficients["Theil Sen"].loc["x + A", "coeff.A"], 2
                ),
                0.5,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_theil_sen._coefficients["Theil Sen"].loc[
                        "x + A", "i.Intercept"
                    ],
                    2,
                ),
                0,
            )

    def test_bayesian_ulr(self):

        test_data_x, test_data_y = test_data()
        test_bayesian = Calibration.Calibration(test_data_x, test_data_y)
        test_bayesian.bayesian()
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x", "coeff.x"
                    ],
                    2,
                ),
                1,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x", "i.Intercept"
                    ],
                    2,
                ),
                0,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x", "coeff.x"
                    ],
                    2,
                ),
                1,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x", "i.Intercept"
                    ],
                    2,
                ),
                0,
            )

    def test_bayesian_mlr(self):

        test_data_x, test_data_y = test_data()
        test_bayesian = Calibration.Calibration(test_data_x, test_data_y)
        test_bayesian.bayesian(["A"])
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x + A", "coeff.x"
                    ],
                    2,
                ),
                0.5,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x + A", "coeff.A"
                    ],
                    2,
                ),
                0.5,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x + A", "i.Intercept"
                    ],
                    2,
                ),
                0,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x + A", "sd.x"
                    ],
                    1,
                ),
                0,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x + A", "sd.A"
                    ],
                    1,
                ),
                0,
            )
        with self.subTest():
            self.assertEqual(
                round(
                    test_bayesian._coefficients["Bayesian (Gaussian)"].loc[
                        "x + A", "sd.Intercept"
                    ],
                    1,
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
