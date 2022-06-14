"""
"""

__author__ = "Idris Hayward"
__copyright__ = "2021, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.4"
__maintainer__ = "Idris Hayward"
__email__ = "CaderIdrisGH@outlook.com"
__status__ = "Beta"

import datetime as dt

from influxdb_client import InfluxDBClient


class InfluxQuery:
    """ Queries and formats data from InfluxDB 2.x database

    Attributes:
        _config (dict): The config file passed in via keyword argument during
        initialisation

        _client (InfluxDBClient): Object that handles connection to InfluxDB
        2.x database

        _query_api (InfluxDBClient.query_api): Handles queries to InfluxDB 2.x
        database

        _measurements (dict): Measurements and timestamps from query

    Methods:
        data_query: Queries the InfluxDB database for the specified
        measurements and stores them in the measurements instance
    """
    def __init__(self, config):
        """Initialises class 

        Keyword Arguments:
            config (dict): Keys correspond to location and access info for
            InfluxDB 2.x database. Keys are:
                IP: IP/URL of database, localhost if on same machine
                Port: Port for database
                Token: Authorisation token to access database
                Organisation: Organisation of auth token
            corresponding organisation

        """
        self._config = config
        self._client = InfluxDBClient(
                url=f"{config['IP']}:{config['Port']}",
                token=config['Token'],
                org=config['Organisation'],
                timeout=15000000
                )
        self._query_api = self._client.query_api()
        self._measurements = {
                "Values": list(),
                "Timestamps": list()
                }

    def data_query(self, query):
        """ Sends flux query, receives data and sorts it in to the measurement
        dict.

        Keyword Arguments:
            query (str): Flux query
        """
        query_return = self._query_api.query(
                query=query,
                org=self._config['Organisation']
                )
        # query_return should only have one table so this just selects the
        # first one
        if len(query_return) > 0:
            self._measurements[
                    'Name'
                    ] = query_return[0].records[0].values['result']
            for record in query_return[0].records:
                values = record.values
                self._measurements['Timestamps'].append(values['_time'])
                self._measurements['Values'].append(values['_value'])
        else:
            self._measurements = None

    def return_measurements(self):
        """ Returns the measurements downloaded from the database

        Returns:
            Copy of self._measurements (dict)
        """
        if self._measurements is not None:
            return self._measurements.copy()
        else:
            return None

    def clear_measurements(self):
        """ Clears measurements downloaded from database

        Returns:
            None 
        """
        self._measurements = {
                "Values": list(),
                "Timestamps": list()
                }


class FluxQuery:
    """Generates flux query for InfluxDB 2.x database

    InfluxDB 2.x uses the flux query language to query metadata and data. This
    class simplifies the query generation process.

    Attributes:
        _query (str): Flux query

    Methods:
        add_field: Adds a field (measurand) to the query

        add_filter: Adds a key and value to the query, all other values that
        the key has are filtered out

        add_group: Adds a group to the query, all measurements are grouped by
        the specified key

        add_window: Adds a window aggregator to the query, measurements will be
        aggregated to specified time windows by the specified function (e.g
        hourly means)

        add_yield: Adds an output function, measurements are output with the
        specified name

        return_query: Returns the query as a string, the query can't be
        accessed outside of the class


    """
    def __init__(self, start, end, bucket, measurement):
        """ Initialises the class instance

        Keyword Arguments:
            start (datetime): Start time of data queried

            end (datetime): End time of data queried

            bucket (str): Bucket where data is stores

            measurement (str): Measurement tag where data is stored
        """
        self._query = (
                f"from(bucket: \"{bucket}\")\n"
                f"  |> range(start: {dt_to_rfc3339(start)}, "
                f"stop: {dt_to_rfc3339(end)})\n"
                f"  |> filter(fn: (r) => r._measurement == "
                f"\"{measurement}\")\n"
                )
        self._start = start
        self._end = end

    def add_field(self, field):
        """ Adds a field to the query

        Keyword arguments:
            field (str): The field to query
        """
        self._query = (
                f"{self._query}  |> filter(fn: (r) => r[\"_field\"] == "
                f"\"{field}\")\n"
                )

    def add_multiple_fields(self, fields):
        """ Adds multiple fields to the query 

        Useful if you want to filter one measurement based on others 

        Keyword Arguments:
            fields (list): All the fields to be queried 
        """
        self._query = (
                f"{self._query}  |> filter(fn: (r) => r[\"_field\"] == "
                f"\"{fields[0]}\" or "
                )
        if len(fields) > 2:
            for field in fields[1:-1]:
                self._query = f"{self._query} r[\"_field\"] == \"{field}\" or "
        self._query = f"{self._query} r[\"_field\"] == \"{fields[-1]}\")\n"

    def add_filter(self, key, value):
        """ Adds a filter to the query

        Keyword Arguments:
            key (str): Key of the tag you want to isolate

            value (str): Tag you want to isolate
        """
        self._query = (
                f"{self._query}  |> filter(fn: (r) => r[\"{key}\"] == "
                f"\"{value}\")\n"
                )

    def add_filter_range(self, field, filter_fields):
        """ Adds filter range to the query

        Adds a filter to the query that only selects measurements when one
        measurement lies inside or outside a specified range

        Keyword arguments:
            field (str): The field that is being filtered 

            filter_fields (list): Contains all fields used to filter field data
        """
        self._query = (
                f"{self._query}  |> keep(columns: [\"_time\", \"_field\", "
                f"\"_value\"])\n"
                f"  |> pivot(rowKey: [\"_time\"], columnKey: [\"_field\"], "
                f"valueColumn: \"_value\")\n"
                )
        for filter_field in filter_fields:
            name = filter_field["Field"]
            min = filter_field["Min"]
            max = filter_field["Max"]
            min_equals_sign = "=" if  filter_field["Min Equal"] else ""
            max_equals_sign = "=" if filter_field["Max Equal"] else ""
            self._query = (
                f"{self._query}"
                f"  |> filter(fn: (r) => r[\"{name}\"] >{min_equals_sign}"
                f" {min} and r[\"{name}\"] <{max_equals_sign} {max})\n"
                )
        self._query = (
                f"{self._query}"
                f"  |> rename(columns: {{\"{field}\": \"_value\"}})\n"
                    )

    def add_group(self, group):
        """ Adds group tag to query

        Keyword Arguments:
            group (str): Key to group measurements by
        """
        self._query = f"{self._query}  |> group(columns: [\"{group}\"])\n"

    def add_window(self, range, function, create_empty=True,
                   time_starting=False, column="_value"):
        """Adds aggregate window to data

        Keyword Arguments:
            range (str): Range of window, use InfluxDB specified ranges
            e.g 1h for 1 hour

            function (str): Aggregate function e.g. mean, median

            create_empty (bool): Add null values where measurements weren't
            made? (default: True)

            time_ending (bool): Timestamp corresponds to end of window?
            (default: True)

            column (str): Column to aggregate (default: "_value")
        """
        time_source = "_stop"
        if time_starting:
            time_source = "_start"
        self._query = (
                f"{self._query}  |> aggregateWindow(every: {range}, "
                f"fn: {function}, column: \"{column}\", timeSrc: "
                f"\"{time_source}\", timeDst: \"_time\", createEmpty: "
                f"{str(create_empty).lower()})\n"
                )

    def keep_measurements(self):
        """ Removes all columns except _time and _value, can help download
        time
        """
        self._query = (
                f"{self._query}  |> keep(columns: [\"_time\", \"_value\"])\n"
                )

    def drop_start_stop(self):
        """ Adds drop function which removes superfluous start and stop
        columns
        """
        self._query = (
                f"{self._query}  |> drop(columns: [\"_start\", \"_stop\"])\n"
                )

    def scale_measurements(self, slope=1, offset=0, power=1, start="", end=""):
        """ Scales measurements. If start or stop is provided in RFC3339
        format, they are scaled within that range only.

        This function uses the map function to scale the measurements, within a
        set range. If a start and/or end range is not provided they default to
        1970-01-01 and 2800-01-01 respectively which should land outside of
        any measurements passed through it for a while at least. This means
        the function only has a limited lifespan, depracating after 2800 but
        in all fairness, we should at least be on Python 4 by then. Feel free
        to fix this future catastrophic bug, Python 4 genii.

        Keyword Arguments:
            slope (int/float): Number to multiply measurements by

            offset (int/float): Number to offset scaled measurements by

            power (int): Exponent to raise the value to before scaling with
            slope and offset

            start (datetime): Start date to scale measurements from
            (Default: None, not added to query)

            end (datetime): End date to scale measurements until
            (Default: None, not added to query)
        """
        if not isinstance(start, dt.datetime):
            start = self._start
        if not isinstance(end, dt.datetime):
            end = self._end
        if isinstance(power, str):
            power = 1
        if slope != 1:
            slope_str = f" * {float(slope)}"
        else:
            slope_str = ""
        if offset != 0:
            off_str = f" + {float(offset)}"
        else:
            off_str = ""
        value_str = "(r[\"_value\"]"
        if power != 1:
            value_str = f"{value_str} ^ {float(power)}"
        value_str = f"{value_str})"
        self._query = (
                f"{self._query}  |> map(fn: (r) => ({{ r with \"_value\": if "
                f"(r[\"_time\"] >= {dt_to_rfc3339(start)} and r[\"_time\"] <= "
                f"{dt_to_rfc3339(end)}) then ({value_str}{slope_str}){off_str}"
                f" else r[\"_value\"]}}))\n"
                )

    def add_yield(self, name):
        """ Adds yield function, allows data to be output

        Keyword Arguments:
            name (str): Name for data, should be unique if multiple queries are
            made
        """
        self._query = f"{self._query}  |> yield(name: \"{name}\")\n"

    def return_query(self):
        """ Returns the query string, _query cannot be called outside of class

        Returns:
            String corresponding to a flux query
        """
        return self._query

def dt_to_rfc3339(input, use_time=True):
    """ Converts datetime to RFC3339 string

    Keyword Arguments:
        input (datetime): Datetime object to convert

        use_time (boolean): Include time? (default: True)

    Returns:
        RFC3339 string converted from input
    """
    if use_time:
        return input.strftime("%Y-%m-%dT%H:%M:%SZ")
    return input.strftime("%Y-%m-%d")
