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
from pathlib import Path
import re
import sqlite3 as sql
from typing import Any, Optional, Union

from calidhayte import Coefficients, Results, Summary, Graphs
from haytex import Report
import numpy as np
import pandas as pd

from modules.idristools import get_json, parse_date_string, all_combinations
from modules.idristools import DateDifference, make_path, file_list
from modules.idristools import folder_list, debug_stats
from modules.influxquery import InfluxQuery, FluxQuery
from modules.grapher import GradGraphs


def read_sqlite(name: str, path: str, tables: list[str] = list()) -> dict[str, pd.DataFrame]:
    """
    Read measurements from a sqlite3 file
    
    Parameters
    ----------
    name : str
        The name of the sqlite3 file
    path : str
        Directory where sqlite3 file is
    tables : str, optional
        Names of tables to import. If unused, imports all in sqlite file
    
    Returns
    -------
    dict containing all tables imported as dataframes, keys represent tables,
    each subdict is a dict with integer keys representing separate folds
    """
    data: dict[str, pd.DataFrame] = dict()
    con = sql.connect(f'{path}/{name}.db')
    if not tables:
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables_to_dl = cursor.fetchall()
        cursor.close()
    else:
        tables_to_dl = tables.copy()
    for table in tables_to_dl:
        sqlite_table = pd.read_sql(
            sql=f"SELECT * from '{table[0]}'",
            con=con
        )
        data[table[0]] = sqlite_table
    con.close()
    return data


def write_sqlite(
        name: str,
        path: str,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        table: Optional[str] = None,
        keep_index: bool = True
        ):
    """
    Write data to sqlite3 file
    
    Paameters
    ---------
    name : str
        The name of the sqlite3 file to be saved
    path : str
        Directory where sqlite3 file is to be saved
    data : pd.DataFrame, dict
        Data to save to sqlite file
    table : str, optional
        Name of table. Not necessary if dict passed as keys used.
        Default is None.
    keep_index : bool
        Save the dataframe index to the sqlite table
    """
    con = sql.connect(f'{path}/{name}.db')
    if isinstance(data, pd.DataFrame) and table is not None:
        data = {table: data}
    elif isinstance(data, pd.DataFrame) and table is None:
        raise ValueError('If passing single dataframe, table name must be given')
    for table_key, df in data.items():
        df.to_sql(
                name=str(table_key),
                con=con,
                if_exists="replace",
                index=keep_index
                )
    con.close()


def relpath(path: Path):
    # Quick and easy way to get relative paths for pgf files
    shortened = list(path.parts[2:])
    shortened.insert(0, "..")
    return "/".join(shortened)


def get_measurements_from_influx(
    run_name: str,
    output_path: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    query_config: dict[str, Any],
    run_config: dict[str, Any],
    influx_config: dict[str, Any],
        ):
    """
    """

    date_calculations = DateDifference(start_date, end_date)
    months_to_cover = date_calculations.month_difference()

    # Download measurements from cache
    measurements: dict[str, dict[str, pd.DataFrame]] = dict()
    cache_folder = f"{output_path}{run_name}/Measurements/"
    cached_files = file_list(cache_folder, extension=".db")
    for cached_file in cached_files:
        field_name = str(re.sub(r"(.*?)/|\.\w*$", "", cached_file))
        measurements[field_name] = read_sqlite(cache_folder, field_name)

    # Download measurements from InfluxDB 2.x on a month by month basis
    for month_num in range(0, months_to_cover):
        date_list = None
        start_of_month = date_calculations.add_month(month_num)
        end_of_month = date_calculations.add_month(month_num + 1)
        empty_data_list = list()
        for name, settings in query_config.items():
            for dev_field in settings["Fields"]:
                if measurements.get(dev_field["Tag"]) is not None:
                    if (month_num) == 0 and (
                        not measurements[dev_field["Tag"]][name].empty
                    ):
                        # If measurements were in cache, skip
                        continue
                # Generate flux query
                query = FluxQuery(
                    start_of_month,
                    end_of_month,
                    settings["Bucket"],
                    settings["Measurement"],
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
                        dev_field["Field"], dev_field["Range Filters"]
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
                        scale_end,
                    )
                # Add in window to average measurements over
                query.add_window(
                    run_config["Runtime"]["Averaging Period"],
                    run_config["Runtime"]["Average Operator"],
                    time_starting=dev_field["Hour Beginning"],
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
                        settings["Measurement"],
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
                            scale_end,
                        )
                    # Set averaging window and remove irrelevant columns
                    sec_query.keep_measurements()
                    sec_query.add_window(
                        run_config["Runtime"]["Averaging Period"],
                        run_config["Runtime"]["Average Operator"],
                        time_starting=dev_field["Hour Beginning"],
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
                    inf_measurements[sec_measurement["Tag"]] = sec_measurements[
                        "Values"
                    ]
                    inf_measurements.set_index("Datetime")
                    influx.clear_measurements()
                measurements[dev_field["Tag"]][name] = pd.concat(
                    [measurements[dev_field["Tag"]][name], inf_measurements]
                )
        # If some measurements were downloaded, populate measurements that
        # weren't downloaded with nan
        if date_list is not None and empty_data_list:
            empty_df = pd.DataFrame(
                data={"Datetime": date_list, "Values": [np.nan] * len(date_list)}
            )
            for empty_data in empty_data_list:
                measurements[empty_data[0]][empty_data[1]] = pd.concat(
                    [measurements[empty_data[0]][empty_data[1]], empty_df]
                )
    for field, data in measurements.items():
        write_sqlite(field, cache_folder, data)

    return measurements


def get_coefficients(
    output_path: str,
    run_name: str,
    measurements: dict[str, Any],
    data_settings: dict[str, Any],
    techniques: dict[str, bool],
    bay_families: dict[str, bool]
        ):
    """
    """
    coeffs_folder = f"{output_path}{run_name}/Coefficients"
    coefficients = dict()
    try:
        coeff_dirs = folder_list(coeffs_folder)
    except FileNotFoundError:
        coeff_dirs = list()
    for field_name in coeff_dirs:
        coefficients[field_name] = dict()
        coeff_files = file_list(f"{coeffs_folder}/{field_name}", extension=".db")
        for coeff_file in coeff_files:
            comparison_name = re.sub(r"(.*?)/|\.\w*$", "", coeff_file)
            coefficients[field_name][comparison_name] = dict()
            con = sql.connect(coeff_file)
            cursor = con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                if re.search("Test|Train", table[0]):
                    coefficients[field_name][comparison_name][table[0]] = pd.read_sql(
                        sql=f"SELECT * from '{table[0]}'",
                        con=con,
                        parse_dates={"Datetime": "%Y-%m-%d %H:%M:%S%z"},
                        index_col="Datetime",
                    )
                else:
                    coefficients[field_name][comparison_name][table[0]] = pd.read_sql(
                        sql=f"SELECT * from '{table[0]}'",
                        con=con,
                        parse_dates={"Datetime": "%Y-%m-%d %H:%M:%S%z"},
                        index_col="index",
                    )
            cursor.close()
            con.close()

    # Loop over fields
    for field, dframes in measurements.items():
        comparison_index = 1
        if coefficients.get(field) is None:
            coefficients[field] = dict()
        # Loop over dependent measurements
        device_names = list(dframes.keys())
        for y_dev_index, y_device in enumerate(device_names[:-1], start=1):
            y_dframe = dframes.get(y_device)
            if isinstance(y_dframe, pd.DataFrame):
                # Loop over independent measurements
                for x_device in device_names[y_dev_index:]:
                    comparison_name = f"({comparison_index}) {x_device} vs {y_device}"
                    print(f"{field} {comparison_name}")
                    if any([
                        re.match(f"\\(\\d*\\) {x_device} vs {y_device}", comp)
                        for comp in coefficients[field].keys()]):
                        comparison_index += 1
                        continue
                    x_dframe = dframes.get(x_device)
                    if isinstance(x_dframe, pd.DataFrame):
                        try:
                            comparison = Coefficients(
                                x_data=x_dframe,
                                y_data=y_dframe,
                                split=data_settings["Split"],
                                test_size=data_settings["Test Size"],
                                seed=data_settings["Seed"],
                            )
                        except ValueError:
                            continue
                        cal_techs = {
                            'Ordinary Least Squares': Coefficients.ols,
                            'Ridge': Coefficients.ridge,
                            'LASSO': Coefficients.lasso,
                            'Elastic Net': Coefficients.elastic_net,
                            'LARS': Coefficients.lars,
                            'LASSO LARS': Coefficients.lasso_lars,
                            'Orthogonal Matching Pursuit': Coefficients.orthogonal_matching_pursuit,
                            'RANSAC': Coefficients.ransac,
                            'Theil Sen': Coefficients.theil_sen
                                }
                        dframe_columns = list(x_dframe.columns)
                        if not comparison.valid_comparison:
                            continue
                        comparison_index = comparison_index + 1
                        if len(dframe_columns) > 2:
                            mv_combinations = pd.MultiIndex.from_product(
                                    [[False, True] for _ in dframe_columns[2:]],
                                    names=dframe_columns[2:]
                                    )
                        else:
                            mv_combinations = pd.MultiIndex.from_product(
                                    [[False]]
                                    )
                        for mv_index in mv_combinations:
                            mv_combo = [mv_combinations.names[ind] for ind, val in enumerate(mv_index) if val]
                            for tech_name, func in cal_techs.items():
                                if techniques.get(tech_name, False):
                                    func(comparison, mv_combo)
                            if techniques["Bayesian"]:
                                for family, use in bay_families.items():
                                    # Loop over all bayesian families in config
                                    # and run comparison if value is true
                                    if use:
                                        comparison.bayesian(mv_combo, family)
                        coefficients[field][
                            comparison_name
                        ] = comparison.return_coefficients()
                        # After comparison is complete, save all coefficients
                        # and test/train data to sqlite3 database
                        cache_path = f"{output_path}{run_name}/Coefficients/{field}"
                        make_path(cache_path)
                        write_sqlite(comparison_name, cache_path, coefficients[field][comparison_name])
                        comparison_index = comparison_index + 1
    return coefficients


def get_results(
        run_config,
        coefficients,
        run_name,
        output_path,
        use_full
        ):
    """
    """
    errors = dict()
    error_techniques = run_config["Errors"]
    for field, comparisons in coefficients.items():
        errors[field] = dict()
        for comparison, coeffs in comparisons.items():
            print(comparison)
            errors[field][comparison] = dict()
            techniques = list(coeffs.keys())
            techniques.remove("Test")
            techniques.remove("Train")
            for index_tech, technique in enumerate(techniques):
                errors[field][comparison][technique] = dict()
                results_path = f"{output_path}{run_name}/Results/{field}/{comparison}/{technique}/Results.db"
                if Path(results_path).is_file():
                    con = sql.connect(results_path)
                    cursor = con.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    for table in tables:
                        errors[field][comparison][technique][table[0]] = pd.read_sql(
                            sql=f"SELECT * from '{table[0]}'",
                            con=con,
                            index_col="index"
                        )
                    continue
                print(technique)
                x_name = re.match(
                        r"(\(\d*\) )(?P<device>.*)( vs .*)",
                        comparison
                        ).group('device')
                y_name = re.sub(r".*(?<= vs )", "", comparison)
                result_calculations = Results(
                    coeffs["Train"],
                    coeffs["Test"],
                    coeffs[technique],
                    comparison,
                    x_name=x_name,
                    y_name=y_name
                )
                err_tech_dict = {
                    'Explained Variance Score': Results.explained_variance_score,
                    'Max Error': Results.max,
                    'Mean Absolute Error': Results.mean_absolute,
                    'Root Mean Squared Error': Results.root_mean_squared,
                    'Root Mean Squared Log Error': Results.root_mean_squared_log,
                    'Median Absolute Error': Results.median_absolute,
                    'Mean Absolute Percentage Error': Results.mean_absolute_percentage,
                    'r2': Results.r2,
                    'Mean Poisson Deviance': Results.mean_poisson_deviance,
                    'Mean Gamma Deviance': Results.mean_gamma_deviance,
                    'Mean Tweedie Deviance': Results.mean_tweedie_deviance,
                    'Mean Pinball Deviance': Results.mean_pinball_deviance
                        }
                for tech, func in err_tech_dict.items():
                    if error_techniques.get(tech, False):
                        func(result_calculations)
                make_path(
                    f"{output_path}{run_name}/Results/{field}/{comparison}/{technique}"
                )
                graphs = Graphs(
                    coeffs["Train"],
                    coeffs["Test"],
                    coeffs[technique],
                    x_name=x_name,
                    y_name=y_name
                        )
                if use_full:
                    graphs.linear_reg_plot(
                        f"[{field}] {comparison} ({technique})"
                    )
                    graphs.bland_altman_plot(
                        f"[{field}] {comparison} ({technique})"
                    )
                    graphs.ecdf_plot(f"[{field}] {comparison} ({technique})")
                if index_tech == 0:
                    result_calculations.temp_time_series_plot(
                        f"{output_path}{run_name}/Results/{field}/{comparison}"
                    )
                result_calculations.save_plots(
                    f"{output_path}{run_name}/Results/{field}/{comparison}/{technique}"
                )
                errors[field][comparison][
                    technique
                ] = result_calculations.return_errors()
                # After error calculation is complete, save all coefficients
                # and test/train data to sqlite3 database
                con = sql.connect(
                    f"{output_path}{run_name}/Results/{field}/{comparison}/"
                    f"{technique}/Results.db"
                )
                for dset, dframe in errors[field][comparison][technique].items():
                    dframe.to_sql(
                        name=dset,
                        con=con,
                        if_exists="replace",
                    )
                con.close()
                con = sql.connect(
                    f"{output_path}{run_name}/Results/{field}/{comparison}/"
                    f"{technique}/Coefficients.db"
                )
                coeffs[technique].to_sql(
                    name="Coefficients", con=con, if_exists="replace"
                )
                con.close()
    return errors


def get_summary(
        errors,
        output_path,
        run_name
        ):
    """
    """
    graphs = GradGraphs()
    for field, comparisons in errors.items():
        for comparison, techniques in comparisons.items():
            comparison_dict = dict()
            graph_path = Path(f"{output_path}{run_name}/Results/{field}/{comparison}")
            for technique, datasets in techniques.items():
                if "Calibrated Test Data" in list(datasets.keys()):
                    comparison_dict[technique] = datasets.get(
                            "Calibrated Test Data"
                            )
                else:
                    comparison_dict[technique] = datasets.get(
                        "Calibrated Test Data (Mean)"
                            )
            summary_data = Summary(comparison_dict)
            best_techniques = summary_data.best_performing(summate="key")
            best_variables = summary_data.best_performing(summate="row")
            graphs.bar_chart(best_techniques, name="Techniques")
            graphs.bar_chart(best_variables, name="Variables")
            best_techniques_tab = pd.DataFrame(
                    data={
                        "Technique": list(best_techniques.keys()),
                        "Total": list(best_techniques.values())
                        }
                    ).set_index("Technique")
            best_variables_tab = pd.DataFrame(
                    data={
                        "Variable": list(best_variables.keys()),
                        "Total": list(best_variables.values())}).set_index("Variable")
            con = sql.connect(f"{graph_path.as_posix()}/Summary.db")
            best_techniques_tab.to_sql(name="Techniques", con=con, if_exists="replace")
            best_variables_tab.to_sql(name="Variables", con=con, if_exists="replace")
            con.close()
            for tech, data in summary_data.best_performing(summate='all').items():
                graphs.bar_chart(data, name=f"{tech}/Variables")
                best_variables_tech_tab = pd.DataFrame(data={"Variable": list(data.keys()), "Total": list(data.values())}).set_index("Variable")
                con = sql.connect(f"{graph_path.as_posix()}/{tech}/Summary.db")
                best_variables_tech_tab.to_sql(name="Variables", con=con, if_exists="replace")
                con.close()
            graphs.save_plots(graph_path.as_posix())


def write_report(
        output_path,
        run_name,
        use_full
        ):
    """
    """
    report_folder = Path(f"{output_path}{run_name}/Results")
    report_fields = [subdir for subdir in report_folder.iterdir() if subdir.is_dir()]
    report = Report(title="Graddnodi Analysis", subtitle=run_name)
    # Report
    for field in report_fields:
        # Part
        report.add_part(field.parts[-1])
        comparisons = [subdir for subdir in field.iterdir() if subdir.is_dir()]
        for comparison in comparisons:
            # Chapter
            report.add_chapter(comparison.parts[-1])
            report.add_figure(
                f"{relpath(comparison)}/Time Series.pgf", caption="Time Series Comparison",
            )
            report.clear_page()
            con = sql.connect(f"{comparison.as_posix()}/Summary.db")
            technique_tab = pd.read_sql(
                sql=f"SELECT * FROM 'Techniques'", con=con, index_col="Technique"
            )
            variable_tab = pd.read_sql(
                sql=f"SELECT * FROM 'Variables'", con=con, index_col="Variable"
            )
            con.close()
            report.add_figure(f"{relpath(comparison)}/Techniques.pgf", caption="Times techniques achieved the lowest error")
            report.add_table(technique_tab, caption="Techniques")
            report.clear_page()
            report.add_figure(f"{relpath(comparison)}/Variables.pgf", caption="Times variable combinations achieved the lowest error")
            report.add_table(variable_tab, caption="Variables")
            report.clear_page()
            techniques = [subdir for subdir in comparison.iterdir() if subdir.is_dir()]
            for technique in techniques:
                # Section
                if use_full:
                    report.add_section(technique.parts[-1])
                    datasets = [subdir for subdir in technique.iterdir() if subdir.is_dir()]
                    for dataset in datasets:
                        report.add_subsection(dataset.parts[-1])
                        # Add coefficients
                        report.add_subsubsection("Coefficients")
                        con = sql.connect(f"{technique.as_posix()}/Coefficients.db")
                        coefficients = pd.read_sql(
                            sql="SELECT * FROM 'Coefficients'", con=con, index_col="index"
                        )
                        report.add_table(
                            coefficients, cols=4, caption="Coefficients"
                        )
                        con.close()
                        report.clear_page()
                        # Add results
                        report.add_subsubsection("Results")
                        con = sql.connect(f"{technique.as_posix()}/Results.db")
                        cursor = con.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        for dat in tables:
                            errors = pd.read_sql(
                                sql=f"SELECT * FROM '{dat[0]}'", con=con, index_col="index"
                            )
                            errors.index.names = ['Variables']
                            report.add_subsubsection(dat[0])
                            report.add_table(errors, caption=dat[0], cols=3)
                        cursor.close()
                        con.close()
                        report.clear_page()
                        if "Uncalibrated" in dataset.parts[-1]:
                            continue

                        # x variable linear regression, Bland-Altman and time series
                        x_dir = Path(f"{dataset.as_posix()}/x")
                        if x_dir.is_dir():
                            report.add_subsection("Calibration")
                            report.add_figure(
                                f"{relpath(x_dir)}/Linear Regression.pgf",
                                caption="Linear Regression",
                            )
                            report.clear_page()
                        techniques = [
                            subdir for subdir in technique.iterdir() if subdir.is_dir()
                        ]
                        # BAs
                        if len(techniques) > 1:
                            report.add_subsection("Bland-Altman Comparisons")
                        else:
                            report.add_subsection("Bland-Altman")
                        ba_graphs = list()
                        ba_glob = dataset.glob("**/Bland-Altman.pgf")
                        for graph in ba_glob:
                            ba_graphs.append(relpath(graph))
                        report.add_figure(
                            ba_graphs, caption="Bland-Altman Graphs", cols=2, rows=2
                        )
                        report.clear_page()
                        # eCDFs
                        if len(techniques) > 1:
                            report.add_subsection("eCDF Variable Comparison")
                        else:
                            report.add_subsection("eCDF")
                        ecdf_graphs = list()
                        ecdf_glob = dataset.glob("**/eCDF.pgf")
                        for graph in ecdf_glob:
                            ecdf_graphs.append(relpath(graph))
                        report.add_figure(
                            ecdf_graphs, caption="eCDF Graphs", cols=2, rows=2
                        )
                        report.clear_page()
    path_to_save = Path(f"{output_path}{run_name}/Report")
    path_to_save.mkdir(parents=True, exist_ok=True)
    report.save_tex(f"{path_to_save.as_posix()}", style_file="Settings/Style.sty")



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
    output_path = args["output_path"]
    config_path = args["config_path"]
    influx_path = args["influx_path"]
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
            output_path,
            start_date,
            end_date,
            query_config,
            run_config,
            influx_config
            )


    # Begin calibration step
    data_settings = run_config["Calibration"]["Data"]
    c_techniques = run_config["Calibration"]["Techniques"]
    bay_families = run_config["Calibration"]["Bayesian Families"]
    coefficients = get_coefficients(
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
    # SUMMARY STATS
    get_summary(errors, output_path, run_name)

    write_report(output_path, run_name, use_full)


if __name__ == "__main__":
    main()
