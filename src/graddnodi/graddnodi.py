#!/bin/python3
"""
Downloads multiple measurements over a set period of time from an InfluxDB
2 database and compares them all in a large collocation study with a range
of regression methods.
"""

__author__ = "Idris Hayward"
__copyright__ = "2023, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "1.0pre"
__maintainer__ = "Idris Hayward"
__email__ = "CaderIdrisGH@outlook.com"
__status__ = "Pre-release"

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
import re
import sqlite3 as sql
from typing import Any, Optional
from typing import TypeAlias, TypedDict, Union

from caderidflux import InfluxQuery
from calidhayte.calibrate import Calibrate
from calidhayte.results import Results
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import tomli

level = logging.DEBUG if os.getenv("PYLOGDEBUG") else logging.INFO
logger = logging.getLogger()
logger.setLevel(level)

formatter = (
    "%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
    if os.getenv("PYLOGDEBUG")
    else "%(message)s"
)

handler = logging.StreamHandler()
handler.setLevel(level)
formatter = logging.Formatter()
handler.setFormatter(formatter)
logger.addHandler(handler)


CalConfDict = TypedDict(
    "CalConfDict",
    {
        "Folds": int,
        "Stratification Groups": int,
        "Seed": int,
        "Scalers": Union[str, list[str]],
    },
)

TechDict = TypedDict(
    "TechDict", {"Default": dict[str, bool], "Random Search": dict[str, bool]}
)

RangeFilterDict = TypedDict(
    "RangeFilterDict",
    {
        "Field": str,
        "Min": Optional[Union[int, float]],
        "Max": Optional[Union[int, float]],
        "Min Equal": Optional[bool],
        "Max Equal": Optional[bool],
    },
)

ScalingDict = TypedDict(
    "ScalingDict",
    {
        "Start": Optional[dt.datetime],
        "End": Optional[dt.datetime],
        "Slope": Optional[Union[int, float]],
        "Offset": Optional[Union[int, float]],
    },
)

FieldDict = TypedDict(
    "FieldDict",
    {
        "Tag": str,
        "Field": str,
        "Boolean Filters": Optional[dict[str, Union[str, float, int, bool]]],
        "Scaling": Optional[ScalingDict],
    },
)

DeviceDict = TypedDict(
    "DeviceDict",
    {
        "Bucket": str,
        "Measurement": str,
        "Boolean Filters": Optional[dict[str, Union[str, float, int, bool]]],
        "Range Filters": list[RangeFilterDict],
        "Fields": list[FieldDict],
        "Hour Beginning": Optional[bool],
        "Use as Reference": Optional[bool],
    },
)

ComparisonConfig = TypedDict(
    "ComparisonConfig",
    {
        "Start": Union[dt.date, dt.datetime],
        "End": Union[dt.date, dt.datetime],
        "Averaging Operator": str,
        "Averaging Period": str,
        "Devices": DeviceDict,
    },
)


CalibrationConfig = TypedDict(
    "CalibrationConfig",
    {
        "Configuration": CalConfDict,
        "Techniques": TechDict,
        "Secondary Variables": Optional[dict[str, list[str]]],
    },
)


class ConfigDict(TypedDict):
    """Expected format of config file."""

    Calibration: CalibrationConfig
    Comparisons: dict[str, ComparisonConfig]
    Metrics: dict[str, bool]


MeasurementsDict: TypeAlias = dict[str, pd.DataFrame]

PipelinesDict: TypeAlias = dict[
    str,
    dict[  # Comparison
        str,
        dict[  # Field
            str,
            dict[  # Technique e.g. Linear Regression
                str,
                dict[  # Scaling (e.g Standard Scaling)
                    str,
                    dict[  # Variables (e.g NO2 + T)
                        int, Union[Pipeline, Path]  # Fold
                    ],
                ],
            ],
        ],
    ],
]

MatchedMeasurementsDict: TypeAlias = dict[
    str, dict[str, dict[str, pd.DataFrame]]  # Comparison  # Field  # x or y
]

ResultsDict: TypeAlias = pd.DataFrame


class GraddnodiResults(TypedDict):
    Measurements: MeasurementsDict
    MatchedMeasurements: MatchedMeasurementsDict
    Pipelines: PipelinesDict
    Results: ResultsDict


def get_json(path: Union[Path, str]) -> ConfigDict:
    """Find json file and returns it as dict.

    Creates blank file with required keys at path if json file is not
    present

    Parameters
    ----------
    path : Path, str
        Path to the json file, including filename and .json extension

    Returns
    -------
    dict[str, str]
        Representing contents of json file

    Raises
    ------
    FileNotFoundError
        If file is not present, blank file created
    ValueError
        If file can not be parsed

    """
    try:
        with open(path) as json_file:
            try:
                return json.load(json_file)
            except json.decoder.JSONDecodeError:
                raise ValueError(
                    f"{path} is not in the proper"
                    f"format. If you're having issues, consider"
                    f"using the template from the Github repo or "
                    f" use the format seen in README.md"
                )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} could not be found, use the "
            f"template from the Github repo or use the "
            f"format seen in README.md"
        )


def import_toml(path: Union[str, Path]) -> ConfigDict:
    """
    Imports toml file and runs type check to ensure it matches config
    specifications

    Parameters
    ----------
    path : Path, list of Path
        Path to the toml file

    Returns
    -------
    ConfigDict
        Configuration parsed from toml file
    """
    if isinstance(path, str):
        path = Path(path)
    with path.open("rb") as toml:
        config = tomli.load(toml)
    type_check_config(config)
    return config


def type_check_config(config: dict[str, Any]) -> None:
    """Check toml config for errors and logs warnings and errors to logger
    object.

    Parameters
    ----------
    config : ConfigDict
        Configuration file to be tested

    Raises
    ------
    ValueError
        If errors are present in config

    """
    errors = []
    warnings = []

    expected_root_keys = ["Calibration", "Comparisons", "Metrics"]
    # Test root
    for key in filter(lambda x: x not in config.keys(), expected_root_keys):
        errors.append(f"Key not found in config: {key}")
    for key in filter(lambda x: x not in expected_root_keys, config.keys()):
        warnings.append(f"Unexpected key in config: {key}")

    expected_calibration_keys = [
        "Configuration",
        "Techniques",
        "Secondary Variables",
    ]
    # Test Calibration subdict
    for key in filter(
        lambda x: x not in config["Calibration"].keys(),
        expected_calibration_keys,
    ):
        errors.append(f"Key not found in Calibration entry: {key}")
    for key in filter(
        lambda x: x not in expected_calibration_keys,
        config["Calibration"].keys(),
    ):
        warnings.append(f"Unexpected key in Calibration Entry: {key}")

    expected_cal_conf = {
        "Folds": int,
        "Stratification Groups": int,
        "Seed": int,
        "Scalers": (str, list),
    }
    # Test Calibration-Configuration
    for key in filter(
        lambda x: x not in config["Calibration"]["Configuration"].keys(),
        expected_cal_conf.keys(),
    ):
        errors.append(
            f"Key not found in Calibration.Configuration entry: {key}"
        )
    for key in filter(
        lambda x: x not in expected_cal_conf.keys(),
        config["Calibration"]["Configuration"].keys(),
    ):
        warnings.append(
            f"Unexpected key in Calibration.Configuration Entry: {key}"
        )
    for key, val in expected_cal_conf.items():
        if config["Calibration"]["Configuration"].get(key) is not None:
            if not isinstance(
                config["Calibration"]["Configuration"].get(key), val
            ):
                errors.append(
                    f"Calibration.Configuration.{key} is type "
                    f"{type(config['Calibration']['Configuration'][key])}, "
                    f"expected {val}"
                )

    # Test Calibration-Techniques
    expected_caltech_keys = ["Default", "Random Search"]
    for key in filter(
        lambda x: x not in config["Calibration"]["Techniques"].keys(),
        expected_caltech_keys,
    ):
        errors.append(f"Key not found in Calibration.Techniques entry: {key}")
    for key in filter(
        lambda x: x not in expected_caltech_keys,
        config["Calibration"]["Techniques"].keys(),
    ):
        warnings.append(
            f"Unexpected key in Calibration.Techniques Entry: {key}"
        )
    for key in filter(
        lambda x: x in expected_caltech_keys,
        config["Calibration"]["Techniques"].keys(),
    ):
        for entry, val in config["Calibration"]["Techniques"][key].items():
            if not isinstance(val, bool):
                errors.append(
                    f"Expected boolean for Calibration.Techniques.{key}."
                    f"{entry}, received {type(val)} instead"
                )

    # Test Calibration-Errors
    for key, val in config["Metrics"].items():
        if not isinstance(val, bool):
            errors.append(
                f"Expected boolean for Metrics.{key}, received "
                f"{type(val)} instead"
            )
    # Test Calibration-Secondary Variables
    for key, val in config["Calibration"]["Secondary Variables"].items():
        if not isinstance(val, list):
            errors.append(
                f"Expected list for Calibration.Secondary Variables.{key}, "
                f"received {type(val)} instead"
            )

    # Test Comparisons subdict
    expected_comp = {
        "Start": (dt.date, dt.datetime),
        "End": (dt.date, dt.datetime),
        "Averaging Operator": str,
        "Averaging Period": str,
        "Devices": dict,
    }

    expected_device = {"Bucket": str, "Measurement": str, "Fields": list}

    optional_device = {
        "Boolean Filters": dict,
        "Range Filters": list,
        "Hour Beginning": bool,
        "Secondary Fields": list,
        "Use as Reference": bool,
    }

    expected_fields = {"Tag": str, "Field": str}

    optional_fields = {"Boolean Filters": dict, "Scaling": list}

    bool_filter_types = (int, float, str, bool)

    range_filter_types = {
        "Field": str,
        "Min": (int, float),
        "Max": (int, float),
        "Min Equal": bool,
        "Max Equal": bool,
    }

    for comp_key, comparison_dict in config["Comparisons"].items():
        for key in filter(
            lambda x: x not in comparison_dict.keys(), expected_comp.keys()
        ):
            errors.append(f"Key not found in Comparisons entry: {key}")
        for key in filter(
            lambda x: x not in expected_comp.keys(), comparison_dict.keys()
        ):
            warnings.append(f"Unexpected key in Comparisons entry: {key}")
        for key, val in expected_comp.items():
            if comparison_dict.get(key) is not None:
                if not isinstance(comparison_dict.get(key), val):
                    errors.append(
                        f"Comparisons.{comp_key}.{key} is type "
                        f"{type(comparison_dict[key])}, expected {val}"
                    )
        # Test Devices
        for device, dev_conf in comparison_dict["Devices"].items():
            for key in filter(
                lambda x: x not in dev_conf.keys(), expected_device.keys()
            ):
                errors.append(
                    f"Key not found in Comparisons.{comp_key}.{device} entry: "
                    f"{key}"
                )
            for key in filter(
                lambda x: x
                not in {**expected_device, **optional_device}.keys(),
                dev_conf.keys(),
            ):
                warnings.append(
                    f"Unexpected key in Comparisons.{comp_key}.{device} entry:"
                    f" {key}"
                )
            # Test device key types
            for conf_key, conf_val in {
                **expected_device,
                **optional_device,
            }.items():
                if not dev_conf.get(conf_key):
                    continue
                if not isinstance(dev_conf.get(conf_key), conf_val):
                    errors.append(
                        f"Comparisons.{comp_key}.Devices.{device}.{conf_key} "
                        f"is type {type(dev_conf[conf_key])}, "
                        f"expected {conf_val}"
                    )
            # Test Fields
            for field in dev_conf.get("Fields", []):
                for key in filter(
                    lambda x: x not in field.keys(), expected_fields.keys()
                ):
                    errors.append(
                        f'Key not found in Comparisons.{comp_key}.{device}.'
                        f'Fields.'
                        f'{field.get("Tag", "(No tag present in field)")}'
                        f'entry: {key}'
                    )
                for key in filter(
                    lambda x: x
                    not in {**expected_fields, **optional_fields}.keys(),
                    field.keys(),
                ):
                    warnings.append(
                        f'Unexpected key in Comparisons.{comp_key}.{device}.'
                        f'Fields.'
                        f'{field.get("Tag", "(No tag present in field)")} '
                        f'entry: {key}'
                    )
                for field_key, field_val in {
                    **expected_device,
                    **optional_device,
                }.items():
                    if not field.get(field_key):
                        continue
                    if not isinstance(field.get(field_key), field_val):
                        errors.append(
                            f'Comparisons.{comp_key}.{device}.Fields.'
                            f'{field.get("Tag", "(No tag present in field)")}.'
                            f'{field_key} is type '
                            f'{type(field.get(field_key))}, expected '
                            f'{field_val}'
                        )
                # Test field boolean filters
                for bool_name, bool_val in field.get(
                    "Boolean Filters", {}
                ).items():
                    if not isinstance(bool_val, bool_filter_types):
                        errors.append(
                            f'Comparisons.{comp_key}.{device}.Fields.'
                            f'{field.get("Tag", "(No tag present in field)")}.'
                            f'Boolean Filters.{bool_name} is type '
                            f'{type(bool_val)}, expected {bool_filter_types}'
                        )

            # Test Boolean Filters
            for bool_name, bool_val in dev_conf.get(
                "Boolean Filters", {}
            ).items():
                if not isinstance(bool_val, bool_filter_types):
                    errors.append(
                        f"Comparisons.{comp_key}.{device}.Boolean Filters."
                        f"{bool_name} is type {type(bool_val)}, expected "
                        f"{bool_filter_types}"
                    )

            # Test Range Filters
            for index, range_filter in enumerate(
                dev_conf.get("Range Filters", [])
            ):
                if not isinstance(range_filter, dict):
                    errors.append(
                        f"Comparisons.{comp_key}.{device}.Range Filters should"
                        f" be a list of dicts, contains {type(range_filter)}"
                    )
                    continue
                for key in filter(
                    lambda x: x not in range_filter_types.keys(),
                    range_filter.keys(),
                ):
                    warnings.append(
                        f"Unexpected key in Comparisons.{comp_key}.{device}."
                        f"Range Filters.[{index}] entry: {key}"
                    )
                for rf_key, rf_type in range_filter_types.items():
                    if range_filter.get(rf_key) is None:
                        continue
                    if not isinstance(range_filter.get(rf_key), rf_type):
                        errors.append(
                            f"Comparisons.{comp_key}.{device}.Range Filters."
                            f"[{index}].{rf_key} is type "
                            f"{type(range_filter.get(rf_key))}, "
                            f"expected {rf_type}"
                        )

            # Test Secondary Fields
            for field in dev_conf.get("Secondary Fields", []):
                for key in filter(
                    lambda x: x not in field.keys(), expected_fields.keys()
                ):
                    errors.append(
                        f'Key not found in Comparisons.{comp_key}.{device}.'
                        f'Secondary Fields.'
                        f'{field.get("Tag", "(No tag present in field)")}'
                        f' entry: {key}'
                    )
                for key in filter(
                    lambda x: x
                    not in {**expected_fields, **optional_fields}.keys(),
                    field.keys(),
                ):
                    warnings.append(
                        f'Unexpected key in Comparisons.{comp_key}.{device}.'
                        f'Secondary Fields.'
                        f'{field.get("Tag", "(No tag present in field)")}'
                        f' entry: {key}'
                    )
                for field_key, field_val in {
                    **expected_device,
                    **optional_device,
                }.items():
                    if not field.get(field_key):
                        continue
                    if not isinstance(field.get(field_key), field_val):
                        errors.append(
                            f'Comparisons.{comp_key}.{device}.'
                            f'Secondary Fields.'
                            f'{field.get("Tag", "(No tag present in field)")}.'
                            f'{field_key} is type {type(field.get(field_key))}'
                            f', expected {field_val}'
                        )
                # Test field boolean filters
                for bool_name, bool_val in field.get(
                    "Boolean Filters", {}
                ).items():
                    if not isinstance(bool_val, bool_filter_types):
                        errors.append(
                            f'Comparisons.{comp_key}.{device}.Secondary '
                            f'Fields.'
                            f'{field.get("Tag", "(No tag present in field)")}'
                            f'.Boolean Filters.{bool_name} is type '
                            f'{type(bool_val)}, expected {bool_filter_types}'
                        )

    for warning in warnings:
        logger.warning(warning)

    for error in errors:
        logger.error(error)

    if errors:
        error_text = "There were errors when parsing the config file"
        raise ValueError(error_text)


def download_cache(path: Union[str, Path]) -> GraddnodiResults:
    """Read saved results from a previous run of Graddnodi.

    Parameters
    ----------
    path : str, Path
        Path to the output folder

    Returns
    -------
    dictionary containing all saved outputs of Graddnodi

    """
    if isinstance(path, str):
        path = Path(path)
    measurements = {}
    pipelines = {}
    results = {}

    # Import measurements
    logger.debug("Locating previously saved measurements")
    measurement_path = path / "Measurements"
    measurement_path.mkdir(parents=True, exist_ok=True)
    measurement_db = measurement_path / "Measurements.db"
    if measurement_db.exists() and not measurement_db.is_dir():
        logger.debug(
            "Found previously saved results in "
            f'{"/".join(measurement_db.parts)}'
        )
        con = sql.connect(measurement_db)
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        cursor.close()
        for table in tables:
            logger.debug(f"Importing measurements for {table[0]} from db")
            data = pd.read_sql(sql=f"SELECT * from '{table[0]}'", con=con)
            if data.shape[0] == 0:
                continue
            data["_time"] = pd.to_datetime(data["_time"])
            data = data.set_index("_time")
            measurements[table[0]] = data
        con.close()
    else:
        logger.debug("No previously saved measurements were found")

    # Import comparisons
    mm_folder = path / "Matched Measurements"
    mm_folder.mkdir(parents=True, exist_ok=True)
    matched_measurements = {}
    for comparison in filter(lambda x: x.is_dir(), mm_folder.glob("**/*")):
        matched_measurements[comparison.parts[-1]] = {}
        for field in filter(lambda x: x.is_file(), comparison.glob("*.db")):
            find_filename = re.match(r"(?P<filename>.*)\.db", field.parts[-1])
            filename = find_filename.group("filename")
            matched_measurements[comparison.parts[-1]][filename] = {}

            logger.debug(
                f"Importing matched measurements for {filename} from db"
            )
            con = sql.connect(field)
            cursor = con.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
            tables = cursor.fetchall()
            cursor.close()
            for table in tables:
                data = pd.read_sql(sql=f"SELECT * from '{table[0]}'", con=con)
                data["_time"] = pd.to_datetime(data["_time"])
                if data.shape[0] == 0:
                    continue
                data = data.set_index("_time")
                matched_measurements[comparison.parts[-1]][filename][
                    table[0]
                ] = data
            con.close()

    # Import pipelines
    pipelines_folder = path / "Pipelines"
    for pkl in filter(
        lambda x: x.is_file(), pipelines_folder.glob("**/*.pkl")
    ):
        find_filename = re.match(r"(?P<filename>.*)\.pkl", pkl.parts[-1])
        if find_filename is None:
            continue
        filename = find_filename.group("filename")
        pp = pipelines
        for part in pkl.parts[-6:-1]:
            try:
                pp = pp[part]
            except KeyError:
                pp[part] = {}
                pp = pp[part]
        pp[int(filename)] = pkl
        logger.debug(pkl)

    # Import results

    results_path = path / "Results"
    if not results_path.is_dir():
        results_path.mkdir(parents=True)
    results_db = results_path / "Results.db"
    con = sql.connect(results_db)
    try:
        results = pd.read_sql(
            sql=f"SELECT * from 'Results'", con=con
        ).set_index("index")
    except pd.errors.DatabaseError:
        logging.debug("No cached results")
        results = pd.DataFrame(
            columns=[
                "Field",
                "Reference",
                "Calibrated",
                "Technique",
                "Scaling Method",
                "Variables",
            ]
        )
    con.close()

    graddnodi_results: GraddnodiResults = {
        "Measurements": measurements,
        "MatchedMeasurements": matched_measurements,
        "Pipelines": pipelines,
        "Results": results,
    }

    return graddnodi_results


def get_measurements_from_influx(
    start_date: dt.datetime,
    end_date: dt.datetime,
    query_config: dict[str, Any],
    run_config: dict[str, Any],
    influx_config: dict[str, Any],
    measurement_db: Path,
    measurements: MeasurementsDict = {},
) -> MeasurementsDict:
    """ """
    # Download measurements from cache
    logger.debug("Downloading measurements for the following devices:")
    logger.debug(measurements.keys())
    # Download measurements from InfluxDB 2.x on a month by month basis
    for name, settings in query_config.items():
        if name in measurements.keys():
            logger.debug(f"Skipping {name} as already in measurements keys")
            continue
        logger.debug(f"Downloading measurements for {name}")
        fields = {}
        bool_filters = settings.get("Boolean Filters", {})
        range_filters = settings.get("Range Filters", {})
        scaling = []
        for field in settings["Fields"] + settings.get("Secondary Fields", []):
            # Columns and tags to rename to
            fields[field["Field"]] = field["Tag"]
            # Boolean filters for specific columns
            if field.get("Boolean Filters") is not None:
                for key, value in field.get("Boolean Filters").items():
                    bool_filters[key] = {"Value": value, "Col": field["Field"]}
            # Scaling
            if field.get("Scaling") is not None:
                for scale_dict in field.get("Scaling"):
                    sc = scale_dict
                    sc["Field"] = field["Field"]
                    scaling.append(sc)
        logger.debug("Querying database")
        inf = InfluxQuery(**influx_config)
        inf.data_query(
            bucket=settings["Bucket"],
            start_date=start_date,
            end_date=end_date,
            measurement=settings["Measurement"],
            fields=list(fields.keys()),
            groups=[],
            win_range=run_config["Averaging Period"],
            win_func=run_config["Averaging Operator"],
            bool_filters=bool_filters,
            range_filters=range_filters,
            hour_beginning=settings.get("Hour Beginning", False),
            scaling=scaling,
            multiindex=False,
            aggregate=True,
            time_split="day",
        )
        logger.debug(f"Downloaded measurements for {name}")
        measurements[name] = inf.return_measurements().rename(columns=fields)
        con = sql.connect(measurement_db)
        logger.debug(f"Saving measurements for {name}")
        measurements[name].to_sql(name=name, con=con, if_exists="replace")
        con.close()

    return measurements


def comparisons(
    measurements: MeasurementsDict,
    cal_settings: dict[str, Any],
    device_config: dict[str, Any],
    pipelines: PipelinesDict,
    matched_measurements: MatchedMeasurementsDict,
    output_path: Path,
) -> tuple[PipelinesDict, MatchedMeasurementsDict]:
    """ """

    cal_class_config = cal_settings["Configuration"]

    techniques = {
        "Linear Regression": Calibrate.linreg,
        "Ridge Regression": Calibrate.ridge,
        "Ridge Regression (Cross Validated)": Calibrate.ridge_cv,
        "Lasso Regression": Calibrate.lasso,
        "Lasso Regression (Cross Validated)": Calibrate.lasso_cv,
        "Elastic Net Regression": Calibrate.elastic_net,
        "Elastic Net Regression (Cross Validated)": Calibrate.elastic_net_cv,
        "Least Angle Regression": Calibrate.lars,
        "Least Angle Lasso Regression": Calibrate.lars_lasso,
        "Orthogonal Matching Pursuit": Calibrate.omp,
        "Bayesian Ridge Regression": Calibrate.bayesian_ridge,
        "Bayesian Automatic Relevance Detection": Calibrate.bayesian_ard,
        "Tweedie Regression": Calibrate.tweedie,
        "Stochastic Gradient Descent": Calibrate.stochastic_gradient_descent,
        "Passive Aggressive Regression": Calibrate.passive_aggressive,
        "RANSAC": Calibrate.ransac,
        "Theil-Sen Regression": Calibrate.theil_sen,
        "Huber Regression": Calibrate.huber,
        "Quantile Regression": Calibrate.quantile,
        "Decision Tree": Calibrate.decision_tree,
        "Extra Tree": Calibrate.extra_tree,
        "Random Forest": Calibrate.random_forest,
        "Extra Trees Ensemble": Calibrate.extra_trees_ensemble,
        "Gradient Boosting Regression": Calibrate.gradient_boost_regressor,
        "Histogram-Based Gradient Boosting Regression": (
            Calibrate.hist_gradient_boost_regressor
        ),
        "Multi-Layer Perceptron Regression": Calibrate.mlp_regressor,
        "Support Vector Regression": Calibrate.svr,
        "Linear Support Vector Regression": Calibrate.svr,
        "Nu-Support Vector Regression": Calibrate.nu_svr,
        "Gaussian Process Regression": Calibrate.gaussian_process,
        "Isotonic Regression": Calibrate.isotonic,
        "XGBoost Regression": Calibrate.xgboost,
        "XGBoost Random Forest Regression": Calibrate.xgboost_rf,
        "Linear GAM": Calibrate.linear_gam,
        "Expectile GAM": Calibrate.expectile_gam,
    }

    device_names = list(device_config.keys())
    logger.debug(device_names)
    for y_dev_index, y_device in enumerate(device_names[:-1], start=1):
        y_config = device_config.get(y_device)
        if y_config is None or not y_config.get("Use as Reference", True):
            logger.info(f"Skipping {y_device} as ground truth")
            continue
        # Loop over ground truth (dependent) devices
        logger.debug(f"Using {y_device} as ground truth")
        y_dframe = measurements.get(y_device)
        logger.error(pipelines.keys())
        if not isinstance(y_dframe, pd.DataFrame):
            continue
        for x_device in device_names[y_dev_index:]:
            # Loop over devices that need calibration (independent)
            x_dframe = measurements.get(x_device)
            if not isinstance(x_dframe, pd.DataFrame):
                continue
            comparison_name = f"{x_device} vs {y_device}"
            logger.info(f"Comparing {x_device} to {y_device}")
            if pipelines.get(comparison_name) is None:
                pipelines[comparison_name] = {}
            if matched_measurements.get(comparison_name) is None:
                matched_measurements[comparison_name] = {}
            for field, sec_vars in cal_settings["Secondary Variables"].items():
                all_vars = [field, *sec_vars]
                skip_field = True
                if (
                    field not in x_dframe.columns
                    or field not in y_dframe.columns
                ):
                    logger.debug(f"{field} not valid for {comparison_name}")
                    continue
                if pipelines[comparison_name].get(field) is None:
                    pipelines[comparison_name][field] = {}
                try:
                    for method_config in ["Default", "Random Search"]:
                        techniques_to_use = cal_settings["Techniques"][
                            method_config
                        ]
                        for technique, method in techniques.items():
                            name = f"{technique} ({method_config})"

                            if not techniques_to_use.get(technique, False):
                                continue
                            if (
                                pipelines[comparison_name][field].get(name)
                                is not None
                            ):
                                continue
                            skip_field = False
                    if skip_field:
                        continue
                    logger.info(f"Beginning {comparison_name} for {field}")
                    calibrate = Calibrate.setup(
                        x_data=x_dframe.loc[
                            :, x_dframe.columns.isin(all_vars)
                        ],
                        y_data=y_dframe.loc[:, [field]],
                        target=field,
                        scaler=cal_class_config["Scalers"],
                        interaction_degree=2,
                        interaction_features=["T", "RH"],
                        vif_bound=None,
                        add_time_column=True,
                        pickle_path=output_path.joinpath(
                            "Pipelines",
                            comparison_name,
                            field
                        ),
                        subsample_data=14170,
                        folds=cal_class_config["Folds"],
                        strat_groups=cal_class_config["Stratification Groups"],
                        seed=cal_class_config["Seed"]
                    )
                except (ValueError, IndexError) as err:
                    logger.exception(
                        "Could not complete %s. "
                        "The indices may not overlap.",
                        comparison_name
                    )
                    continue
                for method_config in ["Default", "Random Search"]:
                    techniques_to_use = cal_settings["Techniques"][
                        method_config
                    ]
                    for technique, method in techniques.items():
                        name = f"{technique} ({method_config})"

                        if not techniques_to_use.get(technique, False):
                            logger.debug("Skipping %s as not in config", name)
                            continue
                        if (
                            pipelines[comparison_name][field].get(name)
                            is not None
                        ):
                            logger.debug(
                                "Skipping %s as pipelines already present",
                                name
                            )
                            continue
                        pipelines[comparison_name][field][name] = {}
                        logger.debug(f"Calibrating using {name}")
                        method(
                            calibrate,
                            name=name,
                            random_search=(method_config == "Random Search"),
                        )
                models = calibrate.return_models()
                pipelines[comparison_name][field].update(models)
                calibrate.clear_models()
                if matched_measurements[comparison_name].get(field) is None:
                    matched_measurements[comparison_name][
                        field
                    ] = calibrate.return_measurements()
                    matched_measures_path = (
                        output_path / "Matched Measurements" / comparison_name
                    )
                    matched_measures_path.mkdir(parents=True, exist_ok=True)
                    matched_measures_db = matched_measures_path / f"{field}.db"
                    logger.debug(
                        f"Saving matched measurements for {comparison_name} "
                        f"{field}"
                    )
                    con = sql.connect(matched_measures_db)
                    for name in matched_measurements[comparison_name][field]:
                        matched_measurements[comparison_name][field][
                            name
                        ].to_sql(name=name, con=con, if_exists="replace")
                    con.close()
    return pipelines, matched_measurements


def get_results(
    pipeline_dict: PipelinesDict,
    matched_measurements: MatchedMeasurementsDict,
    error_config: dict[str, bool],
    error_db_path: Path,
    errors: pd.DataFrame = pd.DataFrame(),
) -> ResultsDict:
    """ """
    for comparison, fields in pipeline_dict.items():
        x_name = re.match(r"(?P<device>.*)( vs .*)", comparison).group(
            "device"
        )
        y_name = re.sub(r".*(?<= vs )", "", comparison)
        for field, techniques in fields.items():
            logger.debug(f"Testing {field} in {comparison}")
            sub_errors = errors[
                np.logical_and(
                    errors["Reference"] == y_name,
                    errors["Calibrated"] == x_name,
                )
            ]
            sub_errors = sub_errors[sub_errors["Field"] == field]
            try:
                result_calculations = Results(
                    matched_measurements[comparison][field]["x"],
                    matched_measurements[comparison][field]["y"],
                    target=field,
                    models=techniques,
                    errors=errors.loc[
                        (errors["Field"] == field)
                        & (errors["Reference"] == y_name)
                        & (errors["Calibrated"] == x_name)
                    ],
                )
            except KeyError:
                continue
            err_tech_dict = {
                "Explained Variance Score": Results.explained_variance_score,
                "Max Error": Results.max,
                "Mean Absolute Error": Results.mean_absolute,
                "Root Mean Squared Error": Results.root_mean_squared,
                "Root Mean Squared Log Error": Results.root_mean_squared_log,
                "Median Absolute Error": Results.median_absolute,
                "Mean Absolute Percentage Error": (
                    Results.mean_absolute_percentage
                ),
                "r2": Results.r2,
                "Centered Root Mean Squared Error": Results.centered_rmse,
                "Unnbiased Root Mean Squared Error": Results.unbiased_rmse,
                "Mean Bias Error": Results.mbe,
                "Reference IQR": Results.ref_iqr,
                "Reference Mean": Results.ref_mean,
                "Reference Range": Results.ref_range,
                "Reference Standard Deviation": Results.ref_sd,
            }
            for tech, func in err_tech_dict.items():
                if error_config.get(tech, False):
                    logger.debug(f"Calculating {tech}")
                    func(result_calculations)
                else:
                    logger.debug(f"Skipping {tech} as not in config")
            err = result_calculations.return_errors()
            err["Field"] = field
            err["Reference"] = y_name
            err["Calibrated"] = x_name
            errors = (
                pd.concat([errors, err])
                .reset_index(drop=True)
                .drop_duplicates()
            )
            con = sql.connect(error_db_path)
            errors.to_sql("Results", con=con, if_exists="replace")
            con.close()
            logger.debug(errors.shape)
    return errors


def main_cli():
    logger.debug(f"Running Graddnodi in debug mode")
    logger.debug(f"Version: {__version__}")
    logger.debug(f"Status: {__status__}")

    arg_parser = argparse.ArgumentParser(
        prog="Graddnodi",
        description="Imports measurements made as part of a collocation "
        "study from an InfluxDB 2.x database and calibrates them using a "
        "variety of techniques",
    )
    arg_parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        help="Alternate location for config json file (Defaults to "
        "./Settings/config.json)",
        default=os.getenv("GRADDNODI_CONFIG", "Settings/config.toml"),
    )
    arg_parser.add_argument(
        "-i",
        "--influx-path",
        type=str,
        help="Alternate location for influx config json file (Defaults to "
        "./Settings/influx.json)",
        default=os.getenv("GRADDNODI_INFLUX", "Settings/influx.json"),
    )
    arg_parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Where output will be saved",
        default=os.getenv("GRADDNODI_OUTPUT", "Output/"),
    )
    args = vars(arg_parser.parse_args())

    output_path = Path(args["output_path"])
    logger.debug(f"Output path: {output_path}")
    config_path = Path(args["config_path"])
    logger.debug(f"Config path: {config_path}")
    influx_path = Path(args["influx_path"])
    logger.debug(f"Influx token path: {influx_path}")

    # Setup
    run_config = import_toml(config_path)
    influx_config = get_json(influx_path)

    cal_settings = run_config["Calibration"]
    metrics = run_config["Metrics"]

    for run_name, run_settings in run_config["Comparisons"].items():
        logger.info(f"Analysing {run_name}")
        query_config = run_settings["Devices"]

        start_date = run_settings["Start"]
        end_date = run_settings["End"]

        # Import previously saved data
        data = download_cache(output_path / run_name)

        logger.debug(f"Downloading measurements from influxdb for {run_name}")

        measurement_db = (
            output_path / run_name / "Measurements" / "Measurements.db"
        )
        data["Measurements"] = get_measurements_from_influx(
            start_date,
            end_date,
            query_config,
            run_settings,
            influx_config,
            measurement_db,
            data["Measurements"],
        )

        cal_settings = run_config["Calibration"]
        data["Pipelines"], data["MatchedMeasurements"] = comparisons(
            data["Measurements"],
            cal_settings,
            query_config,
            data["Pipelines"],
            data["MatchedMeasurements"],
            output_path / run_name,
        )
        error_db_path = output_path / run_name / "Results" / "Results.db"
        data["Results"] = get_results(
            data["Pipelines"],
            data["MatchedMeasurements"],
            error_config=metrics,
            error_db_path=error_db_path,
            errors=data["Results"],
        )
