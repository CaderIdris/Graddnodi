"""
"""

__author__ = "Idris Hayward"
__copyright__ = "2021, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.2"
__maintainer__ = "Idris Hayward"
__email__ = "j.d.hayward@surrey.ac.uk"
__status__ = "Indev"

import datetime as dt
from collections import defaultdict

from modules.influxquery import InfluxQuery, FluxQuery
from modules.idristools import get_json, parse_date_string

if __name__ == "__main__":
    # Setup
    config = get_json("Settings/config.json")
    start_date = parse_date_string(config["Runtime"]["Start"])
    end_date = parse_date_string(config["Runtime"]["End"])
    query_config = config["Devices"]

    # Download measurements from InfluxDB 2.x
    measurements = defaultdict(list)
    for name, settings in config["Devices"].items():
        # Generate flux query
        query = FluxQuery(
                start_date,
                end_date,
                settings["Bucket"],
                settings["Measurement"]
                )
        for dev_field in settings["Fields"]:
            if len(dev_field["Range Filters"]) == 0:
                query.add_field(dev_field["Field"])
            else:
                all_fields = [dev_field["Field"]]
                for value in dev_field["Range Filters"]:
                    all_fields.append(value["Field"])
                query.add_multiple_fields(all_fields)
            for key, value in dev_field["Boolean Filters"].items():
                query.add_filter(key, value)
            if len(dev_field["Range Filters"]) > 0:
                    query.add_filter_range(
                        dev_field["Field"],
                        dev_field["Range Filters"]
                        )
            query.keep_measurements()
            for scale in dev_field["Scaling"]:
                query.scale_measurements(
                        scale["Slope"],
                        scale["Offset"],
                        scale["Start"],
                        scale["End"]
                        )
            query.add_window(
                    config["Runtime"]["Averaging Period"],
                    config["Runtime"]["Average Operator"],
                    time_starting=dev_field["Hour Beginning"]
                    )
            query.add_yield(config["Runtime"]["Average Operator"])
            # Download from Influx
            influx = InfluxQuery(config["Influx"])
            influx.data_query(query.return_query())
            inf_measurements = influx.return_measurements
            if inf_measurements is not None:
                measurements[dev_field["Tag"]].append(
                        {
                            "Name": name,
                            "Measurements": influx.return_measurements()
                            }
                        )
            else:
                continue


#    query = FluxQuery(
#            dt.datetime(2020, 1, 1),
#            dt.datetime(2020, 2, 1),
#            "Nova PM",
#            "Nova PM"
#            )
#    query.add_field("PM2.5")
#    query.add_filter("Serial Number", "C008-0003")
#    query.add_window("1h", "mean")
#    query.drop_start_stop()
#    query.add_yield("data")
#    print(query.return_query())
#    influx = InfluxQuery(settings["Influx"])
#    influx.data_query(query.return_query())

