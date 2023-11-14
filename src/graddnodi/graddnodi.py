#!/bin/python3


""" Downloads multiple measurements over a set period of time from an InfluxDB
2.x database and compares them all in a large collocation study with a range
of regression methods

As part of my PhD, a large collocation study was undertaken at the Bushy Park
air quality monitoring station in Teddington. This program is designed to
import, standardise and compare all the measurements made over the course of
this study and generate a report. It is designed in such a way that any
collocation study can be generated from it if the measurements are stored in
an InfluxDB 2.x database. If the measurements are stored in some other way,
the calibration and report modules can still be used on their own if the data
sent to them is in the correct format.

    Command Line Arguments:
        -c/--config-path (str) [OPTIONAL]: Location of config json. Defaults
        to "Settings/config.json"

        -i/--influx-path (str) [OPTIONAL]: Location of influx condif json. 
        Defaults to "Settings/influx.json"

        -o/--output-path (str) [OPTIONAL]: Where output is saves to. Defaults
        to "Output/"

        -f/--full-output (str) [OPTIONAL]: Generate a full report with scatter,
        eCDF and Bland-Altman plots


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
import logging
import os
from pathlib import Path
import pickle
import re
import sqlite3 as sql
from typing import Any, Optional, Tuple, TypeAlias, TypedDict, Union

from caderidflux import InfluxQuery
from calidhayte.calibrate import Calibrate
from calidhayte.results import Results
from calidhayte.summary import Summary
from calidhayte.graphs import Graphs
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .idristools import get_json, parse_date_string, all_combinations
from .idristools import DateDifference, make_path, file_list
from .idristools import folder_list, debug_stats

level = logging.DEBUG if os.getenv('PYLOGDEBUG') else logging.INFO
logger = logging.getLogger()
logger.setLevel(level)

formatter = \
    '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s' \
    if os.getenv('PYLOGDEBUG') else '%(message)s'

handler = logging.StreamHandler()
handler.setLevel(level)
formatter = logging.Formatter(
)
handler.setFormatter(formatter)
logger.addHandler(handler)

MeasurementsDict: TypeAlias = dict[str, pd.DataFrame]

PipelinesDict: TypeAlias = dict[
    str, dict[  # Comparison
        str, dict[  # Field
            str, dict[  # Technique e.g. Linear Regression
                str, dict[  # Scaling (e.g Standard Scaling)
                    str, dict[  # Variables (e.g NO2 + T)
                        int, Pipeline  # Fold
                            ]
                        ]
                    ]
                ]
            ]
        ]

MatchedMeasurementsDict: TypeAlias = dict[
    str, dict[  # Comparison
        str, dict[  # Field
            str, pd.DataFrame  # x or y
            ]
        ]
    ]

ResultsDict: TypeAlias = pd.DataFrame
        

class GraddnodiResults(TypedDict):
    Measurements: MeasurementsDict
    MatchedMeasurements: MatchedMeasurementsDict
    Pipelines: PipelinesDict
    Results: ResultsDict


def download_cache(path: Union[str, Path]) -> GraddnodiResults:
    """
    Read saved results from a previous run of Graddnodi

    Parameters
    ----------
    path : str, Path
        Path to the output folder

    Returns
    -------
    Dictionary containing all saved outputs of Graddnodi
    """
    if isinstance(path, str):
        path = Path(path)
    measurements = dict()
    pipelines = dict()
    results = dict()

    # Import measurements
    logger.debug('Locating previously saved measurements')
    measurement_path = path / 'Measurements'
    measurement_path.mkdir(parents=True, exist_ok=True)
    measurement_db = measurement_path / 'Measurements.db'
    if measurement_db.exists() and not measurement_db.is_dir():
        logger.debug(
            'Found previously saved results in '
            f'{"/".join(measurement_db.parts)}'
        )
        con = sql.connect(measurement_db)
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        cursor.close()
        for table in tables:
            logger.debug(f'Importing measurements for {table[0]} from db')
            data = pd.read_sql(
                sql=f"SELECT * from '{table[0]}'",
                con=con
            )
            if data.shape[0] == 0:
                continue
            data = data.set_index('_time')
            measurements[table[0]] = data
        con.close()
    else:
        logger.debug('No previously saved measurements were found')

    # Import comparisons
    mm_folder = path / 'Matched Measurements'
    mm_folder.mkdir(parents=True, exist_ok=True)
    matched_measurements = dict()
    for comparison in filter(lambda x: x.is_dir(), mm_folder.glob('**/*')):
        matched_measurements[comparison.parts[-1]] = dict()
        for field in filter(lambda x: x.is_file(), comparison.glob('*.db')):
            find_filename = re.match(r"(?P<filename>.*)\.db", field.parts[-1])
            filename = find_filename.group('filename')
            matched_measurements[comparison.parts[-1]][filename] = dict()

            logger.debug(f'Importing matched measurements for {filename} from db')
            con = sql.connect(field)
            cursor = con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            cursor.close()
            for table in tables:
                data = pd.read_sql(
                    sql=f"SELECT * from '{table[0]}'",
                    con=con
                )
                if data.shape[0] == 0:
                    continue
                data = data.set_index('_time')
                matched_measurements[comparison.parts[-1]][filename][table[0]] = data
            con.close()

    # Import pipelines
    pipelines_folder = path / 'Pipelines'
    for comparison in filter(lambda x: x.is_dir(), pipelines_folder.glob('*/')):
        pipelines[comparison.parts[-1]] = dict()
        logger.debug(f'Looking for pipelines in comparison: {comparison}')
        for field in filter(lambda x: x.is_dir(), comparison.glob('*/')):
            pipelines[comparison.parts[-1]][field.parts[-1]] = dict()
            for technique in filter(lambda x: x.is_dir(), field.glob('*/')):
                pipelines[comparison.parts[-1]][field.parts[-1]][technique.parts[-1]] = dict()
                for scaling in filter(lambda x: x.is_dir(), technique.glob('*/')):
                    pipelines[comparison.parts[-1]][field.parts[-1]][technique.parts[-1]][scaling.parts[-1]] = dict()
                    for variables in filter(lambda x: x.is_dir(), scaling.glob('*/')):
                        pipelines[comparison.parts[-1]][field.parts[-1]][technique.parts[-1]][scaling.parts[-1]][variables.parts[-1]] = dict()
                        for pkl in filter(lambda x: x.is_file(), variables.glob('*.pkl')):
                            find_filename = re.match(r"(?P<filename>.*)\.pkl", pkl.parts[-1])
                            filename = find_filename.group('filename')
                            pipelines[comparison.parts[-1]][field.parts[-1]][technique.parts[-1]][scaling.parts[-1]][variables.parts[-1]][int(filename)] = pkl


    # Import results

    graddnodi_results: GraddnodiResults = {
        "Measurements": measurements,
        "MatchedMeasurements": matched_measurements,
        "Pipelines": pipelines,
        "Results": results
    }

    return graddnodi_results


def get_measurements_from_influx(
    start_date: dt.datetime,
    end_date: dt.datetime,
    query_config: dict[str, Any],
    run_config: dict[str, Any],
    influx_config: dict[str, Any],
    measurement_db: Path,
    measurements: MeasurementsDict = dict(),
        ) -> MeasurementsDict:
    """
    """
    # Download measurements from cache
    logger.debug('Downloading measurements for the following devices:')
    logger.debug(measurements.keys())
    # Download measurements from InfluxDB 2.x on a month by month basis
    for name, settings in query_config.items():
        if name in measurements.keys():
            logger.debug(f'Skipping {name} as already in measurements keys')
            continue
        logger.debug(f'Downloading measurements for {name}')
        fields = dict()
        bool_filters = settings.get("Boolean Filters", {})
        range_filters = settings.get("Range Filters", {})
        scaling = list()
        for field in settings['Fields'] + settings.get("Secondary Fields", []):
            # Columns and tags to rename to
            fields[field['Field']] = field['Tag']
            # Boolean filters for specific columns
            if field.get("Boolean Filters") is not None:
                for key, value in field.get("Boolean Filters").items():
                    bool_filters[key] = {"Value": value, "Col": field['Field']}
            # Scaling
            if field.get("Scaling") is not None:
                for scale_dict in field.get("Scaling"):
                    sc = scale_dict
                    sc['Field'] = field['Field']
                    scaling.append(sc)
        logger.debug('Querying database') 
        inf = InfluxQuery(**influx_config)
        inf.data_query(
            bucket=settings['Bucket'],
            start_date=start_date,
            end_date=end_date,
            measurement=settings['Measurement'],
            fields=list(fields.keys()),
            groups=[],
            win_range=run_config['Averaging Period'],
            win_func=run_config['Averaging Operator'],
            bool_filters=bool_filters,
            range_filters=range_filters,
            hour_beginning=settings.get("Hour Beginning", False),
            scaling=scaling,
            multiindex=False,
            aggregate=True,
            time_split='day'
        )
        logger.debug(f'Downloaded measurements for {name}')
        measurements[name] = inf.return_measurements().rename(columns=fields)
        con = sql.connect(measurement_db)
        logger.debug(f'Saving measurements for {name}')
        measurements[name].to_sql(
            name=name,
            con=con,
            if_exists='replace'
        )
        con.close()

    return measurements


def comparisons(
    measurements: MeasurementsDict,
    cal_settings: dict[str, Any],
    pipelines: PipelinesDict,
    matched_measurements: MatchedMeasurementsDict,
    output_path: Path
        ) -> Tuple[PipelinesDict, MatchedMeasurementsDict]:
    """
    """
    
    cal_class_config = cal_settings['Config']
    cal_method_config = cal_settings['Methods']

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
        "Histogram-Based Gradient Boosting Regression": Calibrate.hist_gradient_boost_regressor,
        "Multi-Layer Perceptron Regression": Calibrate.mlp_regressor,
        "Support Vector Regression": Calibrate.svr,
        "Linear Support Vector Regression": Calibrate.svr,
        "Nu-Support Vector Regression": Calibrate.nu_svr,
        "Gaussian Process Regression": Calibrate.gaussian_process,
        "Isotonic Regression": Calibrate.isotonic,
        "XGBoost Regression": Calibrate.xgboost,
        "XGBoost Random Forest Regression": Calibrate.xgboost_rf
    }

    device_names = list(measurements.keys())
    logger.debug(device_names)
    for y_dev_index, y_device in enumerate(device_names[:-1], start=1):
        # Loop over ground truth (dependent) devices
        logger.debug(f'Using {y_device} as ground truth')
        y_dframe = measurements.get(y_device)
        if not isinstance(y_dframe, pd.DataFrame):
            continue
        for x_device in device_names[y_dev_index:]:
            # Loop over devices that need calibration (independent)
            x_dframe = measurements.get(x_device)
            if not isinstance(x_dframe, pd.DataFrame):
                continue
            comparison_name = f"{x_device} vs {y_device}"
            logger.debug(f'Comparing {x_device} to {y_device}')
            if pipelines.get(comparison_name) is None:
                pipelines[comparison_name] = dict()
            if matched_measurements.get(comparison_name) is None:
                matched_measurements[comparison_name] = dict()
            for field, sec_vars in cal_settings['Comparisons'].items():
                all_vars = [field] + sec_vars
                if field not in x_dframe.columns or field not in \
                    y_dframe.columns:
                    logger.debug(f"{field} not valid for {comparison_name}")
                    continue
                if pipelines[comparison_name].get(field) is None:
                    pipelines[comparison_name][field] = dict()
                try:
                    logger.info(f"Beginning {comparison_name} for {field}")
                    calibrate = Calibrate(
                        x_data=x_dframe.loc[:, x_dframe.columns.isin(all_vars)],
                        y_data=y_dframe.loc[:, [field]],
                        target=field,
                        folds=cal_class_config['Folds'],
                        strat_groups=cal_class_config[
                            'Stratification Groups'
                        ],
                        scaler=cal_class_config['Scalers'],
                        pickle_path = output_path / 'Pipelines' / comparison_name / field,
                        seed=cal_class_config['Seed']
                    )
                except ValueError as err:
                    logger.error(err)
                    logger.error(
                        f"Could not complete {comparison_name}. "
                        f"The indices may not overlap."
                    )
                    continue
                for method_config in ["Default", "Random Search"]:
                    techniques_to_use = cal_settings[f'Techniques ({method_config})']
                    for technique, method in techniques.items():
                        if not techniques_to_use.get(technique, False):
                            logger.debug(f'Skipping {technique} as not in config')
                        name = f"{technique}{f' ({method_config})' if (method_config != 'Default') else ''}"
                        if pipelines[comparison_name][field].get(name) is not None:
                            logger.debug(f"Skipping {technique}")
                            continue
                        pipelines[comparison_name][field][name] = dict()
                        logger.debug(f"Calibrating using {name}")
                        method(calibrate, name=technique, random_search=(method_config == "Random Search"))
                models = calibrate.return_models()
                pipelines[comparison_name][field].update(models)
                calibrate.clear_models()
                matched_measurements[comparison_name][field] = calibrate.return_measurements()
                matched_measures_path = output_path / 'Matched Measurements' / comparison_name
                matched_measures_path.mkdir(parents=True, exist_ok=True)
                matched_measures_db = matched_measures_path / f'{field}.db'
                logger.debug(f'Saving matched measurements for {comparison_name} {field}')
                con = sql.connect(matched_measures_db)
                for name in matched_measurements[comparison_name][field]:
                    matched_measurements[comparison_name][field][name].to_sql(
                        name=name,
                        con=con,
                        if_exists='replace'
                    )
                con.close()
    return pipelines, matched_measurements


def get_results(
    pipeline_dict: PipelinesDict,
    matched_measurements: MatchedMeasurementsDict,
    errors: pd.DataFrame = pd.DataFrame()
        ) -> ResultsDict:
    """
    """
    for comparison, fields in pipeline_dict.items():
        for field, techniques in fields.items():
            logger.debug(f'Testing {field} in {comparison}')
            x_name = re.match(
                    r"(?P<device>.*)( vs .*)",
                    comparison
                    ).group('device')
            y_name = re.sub(r".*(?<= vs )", "", comparison)
            try:
                result_calculations = Results(
                    matched_measurements[comparison][field]['x'],
                    matched_measurements[comparison][field]['y'],
                    target=field,
                    models=techniques
                )
                err_tech_dict = {
                    'Explained Variance Score': Results.explained_variance_score,
                    'Max Error': Results.max,
                    'Mean Absolute Error': Results.mean_absolute,
                    'Root Mean Squared Error': Results.root_mean_squared,
                    #'Root Mean Squared Log Error': Results.root_mean_squared_log,
                    'Median Absolute Error': Results.median_absolute,
                    'Mean Absolute Percentage Error': Results.mean_absolute_percentage,
                    'r2': Results.r2,
                        }
                for tech, func in err_tech_dict.items():
                    func(result_calculations)
                err = result_calculations.return_errors()
                err['Field'] = field
                err['Reference'] = y_name
                err['Calibrated'] = x_name
                errors = pd.concat([errors, err]).reset_index(drop=True)
                logger.debug(errors.shape)
                logger.debug(errors.columns)
            except Exception as err:
                logging.debug(err)
    return errors


def main():
    # Read command line arguments
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
        default="Settings/config.json",
    )
    arg_parser.add_argument(
        "-i",
        "--influx-path",
        type=str,
        help="Alternate location for influx config json file (Defaults to "
        "./Settings/influx.json)",
        default="Settings/influx.json",
    )
    arg_parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Where output will be saved",
        default="Output/",
    )
    arg_parser.add_argument(
            "-f",
            "--full-output",
            action="store_true",
            help="Generate full output"
    )
    args = vars(arg_parser.parse_args())
    output_path = Path(args["output_path"])
    config_path = Path(args["config_path"])
    influx_path = Path(args["influx_path"])
    use_full = args["full_output"]

    # Setup
    run_config = get_json(config_path)
    influx_config = get_json(influx_path)
    run_name = run_config["Runtime"]["Name"]
    query_config = run_config["Devices"]

    start_date = parse_date_string(run_config["Runtime"]["Start"])
    end_date = parse_date_string(run_config["Runtime"]["End"])

    # Download measurements
    measurements = get_measurements_from_influx(
            run_name,
            start_date,
            end_date,
            query_config,
            run_config['Runtime'],
            influx_config
            )


    # Begin calibration step
    data_settings = run_config["Calibration"]["Data"]
    c_techniques = run_config["Calibration"]["Techniques"]
    bay_families = run_config["Calibration"]["Bayesian Families"]
    coefficients = comparisons(
            output_path,
            run_name,
            measurements,
            data_settings,
            c_techniques,
            bay_families
            )


    errors = get_results(
        run_config,
        coefficients,
        run_name,
        output_path,
        use_full
            )
    

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
        default=os.getenv("GRADDNODI_CONFIG", "Settings/config.json")
    )
    arg_parser.add_argument(
        "-i",
        "--influx-path",
        type=str,
        help="Alternate location for influx config json file (Defaults to "
        "./Settings/influx.json)",
        default=os.getenv("GRADDNODI_INFLUX", "Settings/influx.json")
    )
    arg_parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Where output will be saved",
        default=os.getenv("GRADDNODI_OUTPUT", "Output/")
    )
    args = vars(arg_parser.parse_args())

    output_path = Path(args["output_path"])
    logger.debug(f"Output path: {output_path}")
    config_path = Path(args["config_path"])
    logger.debug(f"Config path: {config_path}")
    influx_path = Path(args["influx_path"])
    logger.debug(f"Influx token path: {influx_path}")

    # Setup
    run_config = get_json(config_path)
    influx_config = get_json(influx_path)
    run_name = run_config["Runtime"]["Name"]
    query_config = run_config["Devices"]

    start_date = parse_date_string(run_config["Runtime"]["Start"])
    end_date = parse_date_string(run_config["Runtime"]["End"])

    # Import previously saved data
    data = download_cache(output_path / run_name)
    
    logger.debug('Downloading measurements from influxdb')

    measurement_db = output_path / run_name / 'Measurements' / 'Measurements.db'
    data['Measurements'] = get_measurements_from_influx(
            start_date,
            end_date,
            query_config,
            run_config['Runtime'],
            influx_config,
            measurement_db,
            data['Measurements']
    )

    cal_settings = run_config["Calibration"]
    data['Pipelines'], data["MatchedMeasurements"] = comparisons(
        data['Measurements'],
        cal_settings,
        data['Pipelines'],
        data['MatchedMeasurements'],
        output_path / run_name
    )
#    data['Results'] = get_results(
#        data['Pipelines'],
#        data['MatchedMeasurements']
#    )
#    data['Results'].to_csv('RESULTS.csv')
    




if __name__ == "__main__":
    main()
