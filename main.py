"""
"""

__author__ = "Idris Hayward"
__copyright__ = "2021, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.2"
__maintainer__ = "Idris Hayward"
__email__ = "CaderIdrisGH@outlook.com"
__status__ = "Indev"

import argparse
from collections import defaultdict
import datetime as dt
import pickle

import numpy as np 
import pandas as pd

from modules.idristools import get_json, parse_date_string, all_combinations
from modules.idristools import DateDifference 
from modules.influxquery import InfluxQuery, FluxQuery

def main():
    # Read command line arguments
    arg_parser = argparse.ArgumentParser(
        prog="Graddnodi",
        description="Imports measurements made as part of a collocation "
        "study from an InfluxDB 2.x database and calibrates them using a "
        "variety of techniques"
    )
    measurement_cache_args = arg_parser.add_mutually_exclusive_group()
    measurement_cache_args .add_argument(
        "-m", 
        "--cache-measurements", 
        help="Caches measurements downloaded from InfluxDB 2.x database "
        "in cache folder", 
        action="store_true"
    )
    measurement_cache_args .add_argument(
        "-M", 
        "--use-measurement-cache", 
        help="Uses cached measurements from cache folder", 
        action="store_true"
    )
    results_cache_args = arg_parser.add_mutually_exclusive_group()
    results_cache_args.add_argument(
        "-r", 
        "--cache-results", 
        help="Caches results of calibrations in cache folder", 
        action="store_true"
    )
    results_cache_args.add_argument(
        "-R", 
        "--use-results-cache", 
        help="Uses cached results of calibrations from cache folder", 
        action="store_true"
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
        default="Settings/config.json",
    )
    arg_parser.add_argument(
        "-i",
        "--influx",
        type=str,
        help="Alternate location for influx config json file (Defaults to "
        "./Settings/influx.json)",
        default="Settings/influx.json",
    )
    args = vars(arg_parser.parse_args())
    cache_measurements = args["cache_measurements"]
    use_measurements_cache = args["use_measurement_cache"]
    cache_results = args["cache_results"]
    use_results_cache = args["use_results_cache"]
    cache_path = args["cache_path"]
    config_path = args["config"]
    influx_path = args["influx"]

    # Setup
    run_config = get_json(config_path)
    influx_config = get_json(influx_path)
    run_name = run_config["Runtime"]["Name"]
    start_date = parse_date_string(run_config["Runtime"]["Start"])
    end_date = parse_date_string(run_config["Runtime"]["End"])
    date_calculations = DateDifference(start_date, end_date)
    query_config = run_config["Devices"]
    months_to_cover = date_calculations.month_difference()

    # Download measurements from InfluxDB 2.x on a month by month basis
    if use_measurements_cache:
        with open(f"{cache_path}{run_name}/measurements.pickle", 'rb') as cache:
            measurements = pickle.load(cache)
    else:
        measurements = defaultdict(
                lambda: defaultdict(pd.DataFrame)
                    )
        for month_num in range(0, months_to_cover):
            start_of_month = date_calculations.add_month(month_num)
            end_of_month = date_calculations.add_month(month_num + 1)
            for name, settings in query_config.items():
                for dev_field in settings["Fields"]:
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
                    if not dev_field["Secondary Fields"]:
                        continue
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
                        # If no measurements present, queue them up to be 
                        # populated with nan
                        if sec_measurements is None:
                            influx.clear_measurements()
                            continue
                        inf_measurements[
                                sec_measurement["Tag"]
                                ] = sec_measurements["Values"]
                        influx.clear_measurements()
                    measurements[dev_field["Tag"]][name] = pd.concat(
                        [measurements[dev_field["Tag"]][name], inf_measurements]
                            )
        # Save measurements to a pickle file to be used later. Useful if
        # working offline or using datasets with long query times
        if cache_measurements:
            with open(
                    f"{cache_path}{run_name}/measurements.pickle", 'wb'
                    ) as cache:
                measurements = pickle.dump(measurements, cache)

        # Calibrating measurements against each other

if __name__ == "__main__":
    main()

