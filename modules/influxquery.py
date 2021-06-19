"""
"""

__author__ = "Idris Hayward"
__copyright__ = "2021, Idris Hayward"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.1"
__maintainer__ = "Idris Hayward"
__email__ = "j.d.hayward@surrey.ac.uk"
__status__ = "Indev"

from influxdb_client import InfluxDBClient


class InfluxQuery:
    """ Queries and formats data from InfluxDB 2.x database


    Attributes:
        client (InfluxDBClient): Object that handles connection to InfluxDB
        2.x database

        query_api (InfluxDBClient.query_api): Handles queries to InfluxDB 2.x
        database

    Methods:
    """
    def __init__(self, config):
        """Initialises class 

        Keyword Arguments:
            config (dict): Keys correspond to location and access info for
            InfluxDB 2.x database. Keys are:
                URL: URL of database, localhost if on same machine
                Port: Port for database
                Token: Authorisation token to access database
                Organisation: Organisation of auth token
            corresponding organisation

        """
        self.client = InfluxDBClient(
                url=f"{config['URL']}:{config['Port']}",
                token=config['Token'],
                org=config['Organisation']
                )
        self.query_api = self.client.query_api()


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

    def add_field(self, field):
        """ Adds a field to the query

        Keyword arguments:
            field (str): The field to query
        """
        self._query = (
                f"{self._query}  |> filter(fn: (r) => r._field == "
                f"\"{field}\")\n"
                )

    def add_filter(self, key, value):
        """ Adds a filter to the query

        Keyword Arguments:
            key (str): Key of the tag you want to isolate

            value (str): Tag you want to isolate
        """
        self._query = (
                f"{self._query}  |> filter(fn: (r) => r.{key} == "
                f"\"{value}\")\n"
                )

    def add_group(self, group):
        """ Adds group tag to query

        Keyword Arguments:
            group (str): Key to group measurements by
        """
        self._query = f"{self._query}  |> group(columns: [\"{group}\"])\n"

    def add_window(self, range, function, create_empty=True,
                   time_ending=True, column="_value"):
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
        if not time_ending:
            time_source = "_start"
        self._query = (
                f"{self._query}  |> aggregateWindow(every: {range}, "
                f"fn: {function}, column: \"{column}\", timeSrc: "
                f"\"{time_source}\", timeDst: \"_time\", createEmpty: "
                f"{str(create_empty).lower()})\n"
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
