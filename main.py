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

import datetime as dt
from collections import defaultdict

from modules.influxquery import InfluxQuery, FluxQuery
from modules.idristools import get_json, parse_date_string, all_combinations

if __name__ == "__main__":
    # Setup
    run_config = get_json("Settings/config.json")
    influx_config = get_json("Settings/influx.json")
    start_date = parse_date_string(run_config["Runtime"]["Start"])
    end_date = parse_date_string(run_config["Runtime"]["End"])
    query_config = run_config["Devices"]

    # TODO: REWORK THIS FOR MONTHLY DOWNLOAD WINDOWS FOR YOUR OWN SANITY
    # Download measurements from InfluxDB 2.x
    measurements = defaultdict(list)
    for name, settings in query_config.items():
        print(name)
        for dev_field in settings["Fields"]:
            # Generate flux query
            query = FluxQuery(
                    start_date,
                    end_date,
                    settings["Bucket"],
                    settings["Measurement"]
                    )
            # Check if range filters are present. If yes, a modified query 
            # format needs to be used
            if len(dev_field["Range Filters"]) == 0:
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
            if len(dev_field["Range Filters"]) > 0:
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
                if scale["End"] != "":
                    scale_end = parse_date_string(scale["End"])
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
            inf_measurements = influx.return_measurements
            if inf_measurements is None:
                influx.clear_measurements()
                continue
            measure_dict = {
                    "Name": name,
                    "Measurements": influx.return_measurements()
                    }
            influx.clear_measurements()
            # Add in any secondary measurements such as T or RH
            # Secondary measurements will only have bool filters applied. This
            # ~may~ be controversial and the decision may be reversed but it
            # is what it is for now. Will result in more data present but
            # faster processing times
            if len(dev_field["Secondary Fields"]) > 0:
                measure_dict["Secondary Fields"] = dict()
            for sec_measurement in dev_field["Secondary Fields"]:
                print(sec_measurement["Tag"])
                sec_query = FluxQuery(
                        start_date,
                        end_date,
                        settings["Bucket"],
                        settings["Measurement"]
                        )
                sec_query.add_field(sec_measurement["Field"])
                sec_query.keep_measurements()
                for scale in sec_measurement["Scaling"]:
                    scale_start = ""
                    scale_end = ""
                    if scale["Start"] != "":
                        scale_start = parse_date_string(scale["Start"])
                    if scale["End"] != "":
                        scale_end = parse_date_string(scale["End"])
                    sec_query.scale_measurements(
                            scale["Slope"],
                            scale["Offset"],
                            scale["Power"],
                            scale_start,
                            scale_end
                            )
                for key, value in dev_field["Boolean Filters"].items():
                    sec_query.add_filter(key, value)
                sec_query.add_window(
                        run_config["Runtime"]["Averaging Period"],
                        run_config["Runtime"]["Average Operator"],
                        time_starting=dev_field["Hour Beginning"]
                        )
                query.add_yield(sec_measurement["Tag"])
                influx.data_query(sec_query.return_query())
                sec_measurements = influx.return_measurements
                if sec_measurements is None:
                    influx.clear_measurements()
                    continue
                measure_dict["Secondary Fields"][
                        sec_measurement["Tag"]
                        ] = sec_measurements
            measurements[dev_field["Tag"]].append(measure_dict)
            influx.clear_measurements()

