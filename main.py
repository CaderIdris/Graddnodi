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
        -C/--cache-path (str) [OPTIONAL]: Location of cache folder. Defaults
        to "cache/"

        -c/--config (str) [OPTIONAL]: Location of config json. Defaults to
        "Settings/config.json"

        -i/--influx (str) [OPTIONAL]: Location of influx condif json. Defaults
        to "Settings/influx.json"

"""

__author__ = "Idris Hayward"
__copyright__ = "2021, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.4"
__maintainer__ = "Idris Hayward"
__email__ = "CaderIdrisGH@outlook.com"
__status__ = "Indev"

import argparse
from collections import defaultdict
import datetime as dt
import re
import sqlite3 as sql

import numpy as np 
import pandas as pd

from modules.idristools import get_json, parse_date_string, all_combinations
from modules.idristools import DateDifference, make_path, file_list
from modules.idristools import folder_list
from modules.influxquery import InfluxQuery, FluxQuery
from modules.calibration import Calibration
from modules.errors import Errors 

def main():
    # Read command line arguments
    arg_parser = argparse.ArgumentParser(
        prog="Graddnodi",
        description="Imports measurements made as part of a collocation "
        "study from an InfluxDB 2.x database and calibrates them using a "
        "variety of techniques"
    )
    arg_parser.add_argument(
        "-C",
        "--cache-path",
        type=str,
        help="Location of cache folder",
        default="cache/"
    )
    arg_parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Alternate location for config json file (Defaults to "
        "./Settings/config.json)",
        default="Settings/config.json"
    )
    arg_parser.add_argument(
        "-i",
        "--influx",
        type=str,
        help="Alternate location for influx config json file (Defaults to "
        "./Settings/influx.json)",
        default="Settings/influx.json"
    )
    args = vars(arg_parser.parse_args())
    cache_path = args["cache_path"]
    config_path = args["config"]
    influx_path = args["influx"]

    # Setup
    run_config = get_json(config_path)
    influx_config = get_json(influx_path)
    run_name = run_config["Runtime"]["Name"]
    query_config = run_config["Devices"]

    start_date = parse_date_string(run_config["Runtime"]["Start"])
    end_date = parse_date_string(run_config["Runtime"]["End"])
    date_calculations = DateDifference(start_date, end_date)
    months_to_cover = date_calculations.month_difference()

    # Download measurements from cache
    measurements = defaultdict(
            lambda: defaultdict(pd.DataFrame)
            )
    cache_folder = f"{cache_path}{run_name}/Measurements/"
    cached_files = file_list(cache_folder, extension=".db") 
    for cached_file in cached_files:
        field_name = re.sub(r'(.*?)/|\.\w*$', '', cached_file)
        con = sql.connect(cached_file)
        cursor = con.cursor()
        cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
                )
        tables = cursor.fetchall()
        for table in tables:
            measurements[field_name][table[0]] = pd.read_sql(
                    sql=f"SELECT * from '{table[0]}'",
                    con=con,
                    parse_dates={"Datetime": "%Y-%m-%d %H:%M:%S%z"}
                    )
        cursor.close()
        con.close()

    # Don't cache measurements unless new ones downloaded
    cache_measurements = False 

    # Download measurements from InfluxDB 2.x on a month by month basis
    for month_num in range(0, months_to_cover):
        date_list = None
        start_of_month = date_calculations.add_month(month_num)
        end_of_month = date_calculations.add_month(month_num + 1)
        empty_data_list = list()
        if month_num > 0 and cache_measurements is False:
            break
        for name, settings in query_config.items():
            for dev_field in settings["Fields"]:
                if (month_num) == 0 and (
                        not measurements[dev_field["Tag"]][name].empty
                        ):
                    # If measurements were in cache, skip
                    continue
                cache_measurements = True
                # Generate flux query
                query = FluxQuery(
                        start_of_month,
                        end_of_month,
                        settings["Bucket"],
                        settings["Measurement"]
                        )
                # Check if range filters are present. If yes, a modified
                # query format needs to be used
                if not dev_field["Range Filters"]:
                    query.add_field(dev_field["Field"])
                else:
                    all_fields = [dev_field["Field"]]
                    for value in dev_field["Range Filters"]:
                        all_fields.append(value["Field"])
                    query.add_multiple_fields(all_fields)
                # Adds in any boolean filters (eg Sensor name = X)
                for key, value in dev_field["Boolean Filters"].items():
                    query.add_filter(key, value)
                # Add in any range filters (e.g 0.2 > latitude > 0.1)
                if dev_field["Range Filters"]:
                        query.add_filter_range(
                            dev_field["Field"],
                            dev_field["Range Filters"]
                            )
                # Remove superfluous columns
                query.keep_measurements()
                # Add in scaling
                for scale in dev_field["Scaling"]:
                    scale_start = ""
                    scale_end = ""
                    if scale["Start"] != "":
                        scale_start = parse_date_string(scale["Start"])
                        if scale_start > end_of_month:
                            continue
                    if scale["End"] != "":
                        scale_end = parse_date_string(scale["End"])                        
                        if scale_end < start_of_month:
                            continue
                    query.scale_measurements(
                            scale["Slope"],
                            scale["Offset"],
                            scale["Power"],
                            scale_start,
                            scale_end
                            )
                # Add in window to average measurements over
                query.add_window(
                        run_config["Runtime"]["Averaging Period"],
                        run_config["Runtime"]["Average Operator"],
                        time_starting=dev_field["Hour Beginning"]
                        )
                # Add yield line to return measurements
                query.add_yield(run_config["Runtime"]["Average Operator"])
                # Download from Influx
                influx = InfluxQuery(influx_config)
                influx.data_query(query.return_query())
                inf_measurements = influx.return_measurements()
                if inf_measurements is None:
                    empty_data_list.append((dev_field["Tag"], name))
                    continue 
                inf_measurements.rename({"Values": name})
                if date_list is None:
                    date_list = list(inf_measurements["Datetime"])
                influx.clear_measurements()
                # Add in any secondary measurements such as T or RH
                # Secondary measurements will only have bool filters 
                # applied. This ~may~ be controversial and the decision
                # may be reversed but it is what it is for now. Will 
                # result in more data present but faster processing times
                for sec_measurement in dev_field["Secondary Fields"]:
                    # Generate secondary measurement query
                    sec_query = FluxQuery(
                            start_of_month,
                            end_of_month,
                            settings["Bucket"],
                            settings["Measurement"]
                            )
                    # Add in secondary measurement field
                    sec_query.add_field(sec_measurement["Field"])
                    # Add in boolean filters
                    for key, value in dev_field["Boolean Filters"].items():
                        sec_query.add_filter(key, value)
                    # Add in any scaling
                    for scale in sec_measurement["Scaling"]:
                        scale_start = ""
                        scale_end = ""
                        if scale["Start"] != "":
                            scale_start = parse_date_string(scale["Start"])
                            if scale_start > end_of_month:
                                continue
                        if scale["End"] != "":
                            scale_end = parse_date_string(scale["End"])
                            if scale_end < start_of_month:
                                continue
                        sec_query.scale_measurements(
                                scale["Slope"],
                                scale["Offset"],
                                scale["Power"],
                                scale_start,
                                scale_end
                                )
                    # Set averaging window and remove irrelevant columns
                    sec_query.keep_measurements()
                    sec_query.add_window(
                            run_config["Runtime"]["Averaging Period"],
                            run_config["Runtime"]["Average Operator"],
                            time_starting=dev_field["Hour Beginning"]
                            )
                    query.add_yield(sec_measurement["Tag"])
                    # Query data from database
                    influx.data_query(sec_query.return_query())
                    sec_measurements = influx.return_measurements()
                    # If no measurements present, skip. They will be
                    # populated with nan when dataframe is concatenated
                    if sec_measurements is None:
                        influx.clear_measurements()
                        continue
                    inf_measurements[
                            sec_measurement["Tag"]
                            ] = sec_measurements["Values"]
                    inf_measurements.set_index("Datetime")
                    influx.clear_measurements()
                measurements[dev_field["Tag"]][name] = pd.concat(
                    [measurements[dev_field["Tag"]][name], inf_measurements]
                        )
        # If some measurements were downloaded, populate measurements that
        # weren't downloaded with nan
        if date_list is not None and empty_data_list:
            empty_df = pd.DataFrame(
                data={
                    "Datetime": date_list,
                    "Values": [np.nan] * len(date_list)
                    }
                    )
            for empty_data in empty_data_list:
                measurements[empty_data[0]][empty_data[1]] = (
                            pd.concat(
                                [measurements[empty_data[0]][
                                    empty_data[1]], 
                                    empty_df
                                    ]
                                )
                            )

    # Save measurements to a sqlite3 database be used later. Useful if
    # working offline or using datasets with long query times
    if cache_measurements:
        for tag in measurements.keys():
            measurements[tag] = dict(measurements[tag])
        measurements = dict(measurements)
        if cache_measurements:
            make_path(f"{cache_path}{run_name}/Measurements")
            for field, devices in measurements.items():
                con = sql.connect(
                        f"{cache_path}{run_name}/"
                        f"Measurements/{field}.db"
                    )
                for table, dframe in devices.items():
                    dframe.to_sql(
                            name=table,
                            con=con,
                            if_exists="replace",
                            index=False
                            )
                con.close()

    # Begin calibration step
    device_names = list(query_config.keys())
    data_settings = run_config["Calibration"]["Data"]
    techniques = run_config["Calibration"]["Techniques"]
    bay_families = run_config["Calibration"]["Bayesian Families"]
    coefficients = dict()

    coeffs_folder = f"{cache_path}{run_name}/Coefficients"
    coeff_dirs = folder_list(coeffs_folder)
    for field_name in coeff_dirs:
        coefficients[field_name] = dict()
        coeff_files = file_list(f"{coeffs_folder}/{field_name}", extension=".db")
        for coeff_file in coeff_files:
            comparison_name = re.sub(r'(.*?)/|\.\w*$', '', coeff_file)
            coefficients[field_name][comparison_name] = dict()
            con = sql.connect(coeff_file)
            cursor = con.cursor()
            cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                    )
            tables = cursor.fetchall()
            for table in tables:
                coefficients[field_name][comparison_name][table[0]] = pd.read_sql(
                        sql=f"SELECT * from '{table[0]}'",
                        con=con,
                        parse_dates={"Datetime": "%Y-%m-%d %H:%M:%S%z"}
                        )
            cursor.close()
            con.close()
    cache_coeffs = False

    # Loop over fields
    for field, dframes in measurements.items():
        if coefficients.get(field) is None:
            coefficients[field] = dict() 
        # Loop over dependent measurements
        for index, y_device in enumerate(device_names[:-1]):
            y_dframe = dframes.get(y_device)
            if isinstance(y_dframe, pd.DataFrame):
                # Loop over independent measurements
                for x_device in device_names[index+1:]:
                    comparison_name = f"{x_device} vs {y_device}"
                    if coefficients[field].get(comparison_name) is not None:
                        continue
                    cache_coeffs = True
                    x_dframe = dframes.get(x_device)
                    if isinstance(x_dframe, pd.DataFrame):
                        comparison = Calibration(
                                x_data = x_dframe,
                                y_data = y_dframe,
                                split=data_settings["Split"],
                                test_size=data_settings["Test Size"],
                                seed=data_settings["Seed"]
                                )
                        dframe_columns = list(x_dframe.columns)
                        mv_combinations = [[]]
                        if not comparison.valid_comparison:
                            continue
                        if len(dframe_columns) > 2:
                            mv_combinations.extend(
                                    all_combinations(
                                        dframe_columns[2:]
                                        )
                                    )
                        for mv_combo in mv_combinations:
                            if techniques["Ordinary Least Squares"]:
                                comparison.ols(mv_combo)
                            if techniques["Ridge"]:
                                comparison.ridge(mv_combo)
                            if techniques["LASSO"]:
                                comparison.lasso(mv_combo)
                            if techniques["Elastic Net"]:
                                comparison.elastic_net(mv_combo)
                            if techniques["LARS"]:
                                comparison.lars(mv_combo)
                            if techniques["LASSO LARS"]:
                                comparison.lasso_lars(mv_combo)
                            if techniques["Orthogonal Matching Pursuit"]:
                                if mv_combo: 
                                    # OMP only works with 2 or more independent
                                    # variables
                                    comparison.orthogonal_matching_pursuit(
                                            mv_combo
                                            )
                            if techniques["RANSAC"]:
                                comparison.ransac(mv_combo)
                            if techniques["Theil Sen"]:
                                comparison.theil_sen(mv_combo)
                            if techniques["Bayesian"]:
                                for family, use in bay_families.items():
                                    # Loop over all bayesian families in config
                                    # and run comparison if value is true
                                    if use:
                                        comparison.bayesian(mv_combo, family)

                        coefficients[field][comparison_name
                                ] = comparison.return_coefficients()
                        make_path(f"{cache_path}{run_name}/Coefficients")
                        # After comparison is ocmplete, save all coefficients
                        # and test/train data to sqlite3 database
                        con = sql.connect(
                                f"{cache_path}{run_name}/Coefficients/{field}/"
                                f"{comparison_name}.db"
                                )
                        for dset, dframe in comparison.return_measurements(
                                ).items():
                            dframe.to_sql(
                                    name=dset,
                                    con=con,
                                    if_exists="replace",
                                    index=False
                                    )
                        for comp_technique, dframe in coefficients[field][
                                comparison_name].items():
                            dframe.to_sql(
                                    name=comp_technique,
                                    con=con,
                                    if_exists="replace",
                                    )
                        con.close()

    # Get errors

    for field, comparisons in coefficients.items():
        for comparison, coefficients in comparisons.items():
            techniques = list(coefficients.keys())
            techniques.remove("Test")
            techniques.remove("Train")
            for technique in techniques:
                error_calculations = Errors(
                    coefficients["Train"], 
                    coefficients["Test"], 
                    coefficients[technique]
                    ) 
                error_calculations.explained_variance_score()



if __name__ == "__main__":
    main()

