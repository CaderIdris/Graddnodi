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

import datetime as dt

from modules.influxquery import InfluxQuery, FluxQuery

if __name__ == "__main__":
    query = FluxQuery(
            dt.datetime(2020, 1, 1),
            dt.datetime(2020, 2, 1),
            "Nova PM",
            "Nova PM"
            )
    query.add_group("Serial Number")
    query.add_window("1h", "mean")
    query.add_yield("data")
    print(query.return_query())

