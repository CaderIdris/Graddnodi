from base64 import b64decode
from functools import partial
import json
from io import StringIO
from itertools import cycle
import logging
import os
from pathlib import Path
import tomllib
from typing import Optional

from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from flask import Flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlean as sql

sql.extensions.enable_all()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING if not os.getenv("PYLOGDEBUG", None) \
        else logging.DEBUG,
    format=(
        "%(asctime)s: %(levelname)s - [%(funcName)s]:[%(lineno)d]"
        "- %(message)s"
    )
)

FULL_TABLE_LAYOUT = {"margin": {"pad": 0, "b": 0, "t": 0, "l": 0, "r": 0}}

FIGURE_LAYOUT = {
    "margin": {"pad": 0, "b": 0, "t": 0, "l": 0, "r": 0},
    "showlegend": True,
    "font": {"size": 12},
    "paper_bgcolor": "rgba(255,255,255)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "colorway": [
        "#fb4934",
        "#458588",
        "#98971a",
        "#b16286",
        "#fabd2f",
        "#d65d0e",
        "#7fa2ac",
        "#665c54"
    ],
    "scattermode": "group",
    "legend": {
        "orientation": 'h',
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "bottom",
        "y": 1.02
    }
}

external_stylesheets = [dbc.themes.COSMO]

flask_server = Flask(__name__)
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    server=flask_server
)
server = app.server

output_folder = Path(os.getenv("GRADDNODI_OUTPUT", "Output/"))
logger.debug(
    "Expected output folder (%s) exists: %s",
    output_folder,
    str(output_folder.is_dir())
)
output_options = [
    str(_dir.parts[-1]) for _dir in output_folder.glob("*") if _dir.is_dir()
]
logger.debug("Available options:")
for graddnodi_data_folder in output_options:
    logger.debug("- %s", graddnodi_data_folder)

@callback(
    Output("reference-options", "options"),
    Output("reference-df-index", "data"),
    Input("folder-name", "value")
)
def ref_opt(folders: Optional[list[str]]):
    raw_index = []
    if folders is None:
        logger.debug("No folders selected")
        raise PreventUpdate
    for folder in folders:
        path = output_folder.joinpath(f"{folder}/Results/Results.db")
        logger.debug("Querying sqlite db in %s", path)
        con = sql.connect(path)
        sql_index = pd.read_sql(
            sql=(
                'SELECT DISTINCT "Reference" FROM Results;'
            ),
            con=con,
        )
        sql_index["Folder"] = str(path)
        raw_index.append(
            sql_index
        )
        con.close()
    index = pd.concat(raw_index)
    logger.debug("Number of distinct reference devices: %s", index.shape[0])
    options = [{"label": i, "value": i} for i in index["Reference"].unique()]
    logger.debug(options)
    return options, index.to_json(orient="split")

@callback(
    Output("ref-colours", "data"),
    Input("reference-button-state", "n_clicks"),
    State("reference-options", "value"),
)
def get_colours(
    _: int,
    ref_opts: Optional[list[str]]
):
    """"""
    if not ref_opts:
        raise PreventUpdate
    colors = cycle(
        [
            "#fb4934",
            "#458588",
            "#98971a",
            "#b16286",
            "#fabd2f",
            "#d65d0e",
            "#7fa2ac",
            "#665c54"
        ]
    )
    return {
        k: next(colors) for k in ref_opts
    }


@callback(
    Output("db-index", "data"),
    Input("reference-button-state", "n_clicks"),
    State("reference-df-index", "data"),
    State("reference-options", "value"),
)
def get_df_index(
    _: int,
    reference_index: Optional[str],
    reference_options: Optional[list[str]]
) -> Optional[str]:
    """Get all metadata from sqlite db.

    Parameters
    ----------
    folders : list[str]
        List of all folders containing sqlite db to query data from.

    Returns
    -------
    json representation of dataframe containing all metadata (e.g technique)
    """
    if not reference_index:
        raise PreventUpdate
    ref_df = pd.read_json(StringIO(reference_index), orient="split")
    raw_index = []
    for path, config in ref_df.groupby("Folder"):
        logger.debug("Querying sqlite db in %s", path)
        con = sql.connect(str(path))
        query_opts = {}
        sql_query = [
            (
                'SELECT DISTINCT "Reference", "Field", "Calibrated", '
                '"Technique", "Scaling Method", "Variables", "Fold" FROM '
                'Results '
            )
        ]
        if reference_options:
            logger.debug(reference_options)
            logger.debug(config["Reference"])
            query_opts["ref_opts"] = "', '".join(
                [
                    ref
                    for ref in reference_options
                    if ref in config["Reference"].to_list()
                ]
            )
            sql_query.append('WHERE "Reference" IN (:ref_opts)')
        sql_query.append(";")
        logger.debug("".join(sql_query))
        cursor = con.cursor()
        cursor.execute("".join(sql_query), query_opts)
        sql_index = pd.DataFrame(
            cursor.fetchall(),
            columns=pd.Index(
                [description[0] for description in cursor.description]
            )
        )
        cursor.close()
        sql_index["Database"] = str(path)
        raw_index.append(
            sql_index
        )
        con.close()
    index = pd.concat(raw_index)
    logger.debug("Number of distinct indices: %s", index.shape[0])
    return index.to_json(orient="split")




@callback(
    Output("field-options", "options"),
    Output("calibrated-device-options", "options"),
    Output("technique-options", "options"),
    Output("scaling-options", "options"),
    Output("var-options", "options"),
    Output("fold-options", "options"),
    Output("field-options", "value"),
    Output("calibrated-device-options", "value"),
    Output("technique-options", "value"),
    Output("scaling-options", "value"),
    Output("var-options", "value"),
    Output("fold-options", "value"),
    Input("db-index", "data")
)
def filter_options(
    data: str,
) -> tuple[
    list[dict[str, str]],
    ...
]:
    """Return options to select subset of results.

    Parameters
    ----------
    data : str
        All available parameters.
    """
    levels = [
        "Field",
        "Calibrated",
        "Technique",
        "Scaling Method",
        "Variables",
        "Fold",
    ]
    if not data:
        raise PreventUpdate
    unedited_df: pd.DataFrame = pd.read_json(StringIO(data), orient="split")
    options: dict[str, list[dict[str, str]]] = {}
    for col in levels:
        options[col] = [
            {
                "label": f"{key} [{count}]", "value": key
            } for key, count in
            unedited_df.loc[:, col].sort_index().value_counts().items()
            if not (col == "Technique" and key == "None")
        ]


    return tuple(
        options[col] for col in levels
    ) + ([], [], [], [], [], [])

@callback(
    Output("norm-options", "options"),
    Output("metric-options", "options"),
    Input("reference-button-state", "n_clicks"),
    State("reference-df-index", "data"),
    State("reference-options", "value"),
)
def metric_options(
    _: int,
    reference_index: str,
    reference_options: list[str]
) -> tuple[list[dict[str, str]], ...]:
    """Query valid metrics in results and ref values for normalisation.

    Parameters
    ----------
    _ : int
        Unused, triggers callback.
    reference_index : str
        Reference devices and the database they're found in.
    reference_options : list[str]
        Selected reference devices, use all if empty.

    """
    if not reference_index:
        raise PreventUpdate
    index_cols = [
        "index",
        "Field",
        "Reference",
        "Calibrated",
        "Technique",
        "Scaling Method",
        "Variables",
        "Fold",
        "Count",
    ]
    ref_df = pd.read_json(StringIO(reference_index), orient="split")
    metric_columns = set()
    norm_columns = set()
    for path, config in ref_df.groupby("Folder"):
        logger.debug("Querying sqlite db in %s", path)
        con = sql.connect(str(path))
        query_opts = {}
        sql_query = ['SELECT * FROM "Results" ']
        if reference_options:
            logger.debug(reference_options)
            logger.debug(config["Reference"])
            query_opts["ref_opts"] = "', '".join(
                [
                    ref
                    for ref in reference_options
                    if ref in config["Reference"].to_list()
                ]
            )
            sql_query.append('WHERE "Reference" IN (:ref_opts)')
        sql_query.append(" LIMIT 0;")
        logger.debug("".join(sql_query))
        cursor = con.cursor()
        cursor.execute("".join(sql_query), query_opts)
        columns = [description[0] for description in cursor.description]
        cursor.close()
        norm_columns.update(
            {col for col in columns if col[:10] == "Reference "}
        )
        metric_columns.update(
            {
                col for col in columns
                if col[:10] != "Reference " and col not in index_cols
            }
        )
        con.close()
    norm_options: list[dict[str, str]] = [
        {
            "label": "None",
            "value": ""
        }
    ]

    norm_options.extend(
        [
            {
                "label": i.replace("Reference ", ""),
                "value": i,
            } for i in sorted(norm_columns)
        ]
    )
    metric_options: list[dict[str, str]] = [
        {"label": i, "value": i} for i in sorted(metric_columns)
    ]
    return (
        norm_options,
        metric_options
    )


@callback(
    Output("results-df", "data"),
    Input("submit-button-state", "n_clicks"),
    State("db-index", "data"),
    State("reference-options", "value"),
    State("field-options", "value"),
    State("calibrated-device-options", "value"),
    State("technique-options", "value"),
    State("scaling-options", "value"),
    State("var-options", "value"),
    State("fold-options", "value"),
    State("split-by-options", "value"),
    State("norm-options", "value"),
    State("metric-options", "value"),
    State("remove-outliers", "value"),
)
def get_results_df(
    _: int,
    index_to_query: Optional[str],
    ref_opts: list[str],
    field_opts: list[str],
    cal_opts: list[str],
    tech_opts: list[str],
    scaling_opts: list[str],
    var_opts: list[str],
    fold_opts: list[str],
    split_opts: list[str],
    norm_opt: str,
    metric_opt: str,
    outlier_opt: str
) -> dict[str, str]:

    """Query results from database.

    Parameters
    ----------
    index_to_query : str, optional
        Index values to query from results, in json split orientation.

    Returns
    -------
    tuple containing subset of results and results with no calibration.
    """
    no_norm = [
        "Mean Absolute Percentage Error",
        "r2",
        "Explained Variance Score",
    ]
    options = {
        "Reference": ref_opts,
        "Field": field_opts,
        "Calibrated": cal_opts,
        "Technique": tech_opts,
        "Scaling Method": scaling_opts,
        "Variables": var_opts,
        "Fold": fold_opts,
    }
    query_params = {
        "metric": metric_opt,
        "splits": '", "'.join(split_opts),
        "norm": f'"{norm_opt}"' if norm_opt and metric_opt not in no_norm else 1
    }

    if not index_to_query or not split_opts:
        raise PreventUpdate
    index_df = pd.read_json(StringIO(index_to_query), orient="split")
    results = {}
    for database, sub_df in index_df.groupby("Database"):
        db = Path(str(database))
        folder_name = db.parts[-3]
        con = sql.connect(db)
        cursor = con.cursor()
        con.set_trace_callback(logger.debug)
        if outlier_opt == "Yes":
            iqr_query = [
                (
                    f'SELECT "{query_params["splits"]}", '
                    f'percentile("{query_params["metric"]}" / '
                    f'{query_params["norm"]}, 25) AS Q1, '
                    f'percentile("{query_params["metric"]}" / '
                    f'{query_params["norm"]}, 75) AS Q3, '
                    f'percentile("{query_params["metric"]}" / '
                    f'{query_params["norm"]}, 75) - '
                    f'percentile("{query_params["metric"]}" / '
                    f'{query_params["norm"]}, 75) as IQR, '
                    f'percentile("{query_params["metric"]}" / '
                    f'{query_params["norm"]}, 25) - '
                    f'(1.5 * (percentile("{query_params["metric"]}" '
                    f'/ {query_params["norm"]}, 75) - '
                    f'percentile("{query_params["metric"]}" / '
                    f'{query_params["norm"]}, 25))) AS Low, '
                    f'percentile("{query_params["metric"]}" / '
                    f'{query_params["norm"]}, 75) + '
                    f'(1.5 * (percentile("{query_params["metric"]}" '
                    f'/ {query_params["norm"]}, 75) - '
                    f'percentile("{query_params["metric"]}" / '
                    f'{query_params["norm"]}, 25))) AS High '
                    'FROM "Results" '
                )
            ]
            for ind_col, vals in options.items():
                if not vals:
                    continue
                filter_by = [
                    val for val in vals if val in sub_df[ind_col].unique()
                ]
                iqr_query.append(
                        f'{"WHERE" if len(iqr_query) == 1 else "AND"} '
                        f'''"{ind_col}" IN ('{"', '".join(filter_by)}') '''
                )
                query_params[ind_col] = filter_by
            iqr_query.append(f'GROUP BY "{query_params["splits"]}";')
            cursor.execute("".join(iqr_query), query_params)
            limits = pd.DataFrame(
                cursor.fetchall(),
                columns=pd.Index([desc[0] for desc in cursor.description])
            )
        else:
            iqr_query = [
                (
                    f'SELECT "{query_params["splits"]}", '
                    f'MIN("{query_params["metric"]}" / '
                    f'{query_params["norm"]}) AS Low, '
                    f'MAX("{query_params["metric"]}" / '
                    f'{query_params["norm"]}) AS High '
                    'FROM "Results" '
                )
            ]
            for ind_col, vals in options.items():
                if not vals:
                    continue
                filter_by = [
                    val for val in vals if val in sub_df[ind_col].unique()
                ]
                iqr_query.append(
                        f'{"WHERE" if len(iqr_query) == 1 else "AND"} '
                        f'''"{ind_col}" IN ('{"', '".join(filter_by)}') '''
                )
                query_params[ind_col] = filter_by
            iqr_query.append(f'GROUP BY "{query_params["splits"]}";')
            cursor.execute("".join(iqr_query), query_params)
            limits = pd.DataFrame(
                cursor.fetchall(),
                columns=pd.Index([desc[0] for desc in cursor.description])
            )

        summary_list = []

        for _, bounds in limits.iterrows():
            normed_col_name = f'"{query_params["metric"]}" / {query_params["norm"]}'
            bounds_query = [
                (
                    f'SELECT "{query_params["splits"]}", '
                    f'median({normed_col_name}) AS Median, '
                    f'(percentile({normed_col_name}, 75) - percentile({normed_col_name}, 25)) AS IQR '
                    'FROM "Results" '

                )
            ]
            for i, ind in enumerate(split_opts):
                bounds_query.append(
                    f'{"WHERE" if i == 0 else "AND"} '
                    f'"{ind}" == \'{bounds[ind]}\' '
                )

            for ind_col, vals in options.items():
                if not vals or ind_col in split_opts:
                    continue
                filter_by = query_params[ind_col]
                bounds_query.append(
                    f'''AND "{ind_col}" IN ('{"', '".join(filter_by)}') '''
                )
            bounds_query.append(
                f'AND "{query_params["metric"]}" / {query_params["norm"]} '
                f'BETWEEN {bounds["Low"]} AND {bounds["High"]}'
            )
            bounds_query.append(";")
            cursor.execute("".join(bounds_query), query_params)
            bounds_df = pd.DataFrame(
                cursor.fetchall(),
                columns=pd.Index([desc[0] for desc in cursor.description])
            )

            summary_query = [
                (
                    f'SELECT "{query_params["splits"]}", '
                    f'AVG({normed_col_name}) AS Mean, '
                    f'stddev({normed_col_name}) AS σ, '  # noqa: RUF001
                    f'min({normed_col_name}) FILTER (WHERE {normed_col_name} > ({bounds_df.loc[0, "Median"]} - (1.5 * {bounds_df.loc[0, "IQR"]}))) AS Min, '
                    f'percentile({normed_col_name}, 25) AS Q1, '
                    f'median({normed_col_name}) AS Median, '
                    f'percentile({normed_col_name}, 75) AS Q3, '
                    f'max({normed_col_name}) FILTER (WHERE {normed_col_name} < ({bounds_df.loc[0, "Median"]} + (1.5 * {bounds_df.loc[0, "IQR"]}))) AS Max, '
                    f'percentile({normed_col_name}, 75) - '
                    f'percentile({normed_col_name}, 25) AS IQR, '
                    f'count({normed_col_name}) AS Count '
                    'FROM "Results" '
                )
            ]
            for i, ind in enumerate(split_opts):
                summary_query.append(
                    f'{"WHERE" if i == 0 else "AND"} '
                    f'"{ind}" == \'{bounds[ind]}\' '
                )

            for ind_col, vals in options.items():
                if not vals or ind_col in split_opts:
                    continue
                filter_by = query_params[ind_col]
                summary_query.append(
                    f'''AND "{ind_col}" IN ('{"', '".join(filter_by)}') '''
                )
            summary_query.append(
                f'AND "{query_params["metric"]}" / {query_params["norm"]} '
                f'BETWEEN {bounds["Low"]} AND {bounds["High"]}'
            )
            summary_query.append(";")
            cursor.execute("".join(summary_query), query_params)
            query_df = pd.DataFrame(
                cursor.fetchall(),
                columns=pd.Index([desc[0] for desc in cursor.description])
            )
            summary_list.append(
                query_df
            )
        if summary_list:
            final_list = pd.concat(summary_list).round(3).set_index(split_opts)
            if len(split_opts) > 1:
                final_list.index = [
                    ", ".join(i) for i in final_list.index.to_list()
                ]
            logger.debug(
                "Generated results for %s: %s",
                folder_name,
                final_list.shape
            )
            results[folder_name] = final_list.to_json(orient='split')
        cursor.close()
        con.close()
    return results


def empty_figure(missing_graph_type: str = ""):
    fig = go.Figure(
        layout={
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            **FULL_TABLE_LAYOUT,
        }
    )
    if missing_graph_type != "":
        fig.add_annotation(
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            text=f"{missing_graph_type} is not implemented",
            showarrow=False,
            font={"size": 32},
        )
    return fig


def result_table_plot(dfs):
    """ """
    table_fig = make_subplots(
        len(dfs),
        1,
        subplot_titles=list(dfs.keys()),
        specs=[
            [{"type": "table"}]
            for _ in dfs
        ]
    )
    size = 0
    for i, (_, data) in enumerate(dfs.items(), start=1):
        size += data.shape[0]
        table_cell_format = [
            ".2f" if i == "float64" else ""
            for i in data.dtypes
        ]
        table_fig.add_trace(
            go.Table(
                header={"values": list(data.columns), "align": "left"},
                cells={
                    "values": data.transpose().to_numpy().tolist(),
                    "align": "left",
                    "format": table_cell_format
                },
            ),
            row=i,
            col=1
        )
    table_fig.update_layout(
        autosize=False,
        width=2100,
        height=800 * (len(dfs)),
    )
    return table_fig


def box_plot(dfs, col, norm_options, colours, legend):
    """ """
    no_norm = [
        "Mean Absolute Percentage Error",
        "r2",
        "Explained Variance Score",
    ]
    box_plot_fig = go.Figure()
    for key, data in dfs.items():
        box_plot_fig.add_trace(
            go.Box(
                x=data.index,
                q1=data["Q1"],
                median=data["Median"],
                q3=data["Q3"],
                lowerfence=data["Min"],
                upperfence=data["Max"],
                mean=data["Mean"],
                sd=data["σ"],
                name=key,
                offsetgroup=key,
                showlegend=legend,
                fillcolor="rgba(0,0,0,0)",
                marker_color = colours[key]
            )
        )

    if norm_options and norm_options != "None" and col not in no_norm:
        title = f"{col} /<br>{norm_options}"
    else:
        title = col

    box_plot_fig.update_layout(
        **FIGURE_LAYOUT,
        width=1200,
        height=800,
        boxmode='group',
        yaxis={
            "autorange": False,
            "title": title,
        },
        xaxis={
                    "tickangle": -45
        }

    )
    return box_plot_fig


@callback(
    Output("summary-figure", "figure"),
    Input("graph-type", "value"),
    Input("results-df", "data"),
    Input("metric-options", "value"),
    State("norm-options", "value"),
    State("ref-colours", "data"),
)
def results_plot(plot_type, df_split, col, norm, colours, legend=True):
    data = {}
    if not df_split:
        raise PreventUpdate
    logger.debug(
        "Plotting %s split by %s keys",
        plot_type,
        len(df_split.keys())
    )
    for study, splits in df_split.items():
        data[study] = pd.read_json(StringIO(splits), orient='split')
    match plot_type:
        case "results-table-tab":
            return result_table_plot(data)
        case "box-plot-tab":
            return box_plot(data, col, norm, colours, legend=legend)
    return empty_figure(plot_type)

def double_stack_config(sub_config, plot_config, key, default=None):
    if default is None:
        default = []
    return plot_config.get(key, sub_config.get(key, default))


@callback(
    Output("graphs-generated-text", "data"),
    Input("generate-graphs-button", "contents"),
    prevent_initial_call=True,
)
def generate_graphs_for_paper(button):
    if not button:
        raise PreventUpdate
    _, content = button.split(',')
    toml_string = b64decode(content).decode()
    tag_dict = {
        "Results Table": "results-table-tab",
        "Box Plot": "box-plot-tab",
        "Violin Plot": "violin-plot-tab",
        "Proportion Improved": "prop-imp-plot-tab",
        "Target Diagram": "target-plot-tab",
    }
    config = tomllib.loads(toml_string)
    graph_path = Path("./Graphs")
    graph_path.mkdir(exist_ok=True, parents=True)
    for run, run_data in config.items():
        config = run_data["Configuration"]
        graphs = run_data["Graphs"]
        filepath = graph_path.joinpath(f"{run}.png")
        if filepath.exists():
            logger.info("Skipping %s", run)
            continue
        logger.info("Generating %s", run)
        plot = make_subplots(
            cols=1,
            rows=len(graphs),
            vertical_spacing=0.01,
            shared_xaxes=True
        )
        for index, data in enumerate(graphs, start=1):
            arg_folders = config["Folders"]
            reference_options, reference_index = ref_opt(arg_folders)
            reference_options = [i["value"] for i in reference_options]
            df_index = get_df_index(
                1,
                reference_index,
                data.get("Reference", reference_options)
            )
            arg_reference = double_stack_config(data, config, "Reference", reference_options)
            arg_field = double_stack_config(data, config, "Fields")
            arg_cal = double_stack_config(data, config, "Calibrated")
            arg_tech = double_stack_config(data, config, "Regression Techniques")
            arg_scaling = double_stack_config(data, config, "Scaling Techniques")
            arg_variables = double_stack_config(data, config, "Variables")
            arg_fold = double_stack_config(data, config, "Folds")
            arg_split = double_stack_config(data, config, "Split By")
            arg_norm = double_stack_config(data, config, "Normalise By", "")
            arg_metric = double_stack_config(data, config, "Metric", "Mean Absolute Error")
            arg_outlier = double_stack_config(data, config, "Remove Outliers", "Yes")
            results = get_results_df(
                1,
                df_index,
                ref_opts = arg_reference,
                field_opts = arg_field,
                cal_opts = arg_cal,
                tech_opts = arg_tech,
                scaling_opts = arg_scaling,
                var_opts = arg_variables,
                fold_opts = arg_fold,
                split_opts = arg_split,
                norm_opt = arg_norm,
                metric_opt = arg_metric,
                outlier_opt = arg_outlier
            )
            colours = get_colours(1, data.get("Reference", reference_options))
            subplot = results_plot(
                tag_dict[data.get("Plot Type", "Box Plot")],
                results,
                data.get("Metric", "Mean Absolute Error"),
                data.get("Normalise By", ""),
                colours,
                legend = index == 1
            )
            traces = subplot.select_traces()
            plot_config = subplot.to_dict()
            layout = plot_config.get("layout", {})
            for trace in traces:
                plot.add_trace(
                    trace,
                    row=index,
                    col=1
                )
            if len(graphs) > 1:
                plot.add_annotation(
                    xref='x domain',
                    yref='y domain',
                    x=0.01,
                    y=0.97,
                    text=f'({chr(96+index)})',
                    font={"size": 24},
                    showarrow=False,
                    row=index,
                    col=1
                )
            plot.update_layout(**{
                k: v
                for k, v in layout.items()
                if k not in ["xaxis", "yaxis"]
            })
            xaxis_config = layout.get('xaxis', {})
            xaxis_config.update(data.get("X Axis", {}))
            xaxis_config.update({
                "ticks": "outside",
                "gridcolor": "#b6bdbf",
                "showgrid": True,
                "showline": True,
                "mirror": True,
                "linecolor": "#1d2021",
                #"title": {"font": {"size": 16}},
                "tickangle": -45
            })
            yaxis_config = layout.get('yaxis', {})
            yaxis_config.update(data.get("Y Axis", {}))
            yaxis_config.update({
                "ticks": "outside",
                "gridcolor": "#b6bdbf",
                "showgrid": True,
                "zeroline": True,
                "showline": True,
                #"title": {"font": {"size": 16}},
                "mirror": True,
                "zerolinecolor": "#b6bdbf",
                "linecolor": "#1d2021",
                "minor": {
                    "ticks": "outside",
                    "gridcolor": "#e5e6e7",
                    "showgrid": True,

                }
            })
            plot.layout[f'xaxis{index}'].update(**xaxis_config)
            plot.layout[f'yaxis{index}'].update(**yaxis_config)
        plot.update_xaxes(**config.get("X Axis", {}))
        plot.update_yaxes(**config.get("Y Axis", {}))
        plot.update_yaxes(
            ticks = "outside",
            gridcolor = "#1d2021",
            showgrid = True,
            zeroline = True,
            zerolinecolor = "#1d2021"
        )
        dimension_dict = {
            "width": 2100,
            "height": 800 * (len(graphs))
        }
        layout_dict = config.get("Layout", {})
        plot.update_layout(
            boxmode='group',
            **FIGURE_LAYOUT,
            **layout_dict,
            **{k: v for k, v in dimension_dict.items() if k not in layout_dict}
        )
        plot.write_image(filepath)
    return None
    for i in []:
        filepath = graph_path.joinpath(f"{run}.png")
        if filepath.exists():
            continue
        _, _, _, _, _, _, cci, _ = filter_options(
            None,
            db_index,
            data.get("Reference Instruments", []),
            data.get("Fields", []),
            data.get("Calibration Devices", []),
            data.get("Techniques", []),
            data.get("Scaling Techniques", []),
            data.get("Variables", []),
            data.get("Folds", []),
        )
        results, raw = get_results_df(
            cci, path, data.get("Reference Instruments", [])
        )
        grouped = grouped_data(
            results,
            data.get("Split By", ["Calibrated"]),
            data.get("Normalise By", "None"),
            data.get("Remove Outliers", "Yes"),
            raw,
        )
        metric_columns = data.get("Metric Column", "r2")
        if isinstance(metric_columns, str):
            plot = results_plot(
                tag_dict.get(data.get("Plot Type", "Box Plot")),
                grouped,
                metric_columns,
                data.get("Normalise By", "None"),
            )
            plot.update_xaxes(**data.get("X Axis", {}))
            plot.update_yaxes(**data.get("Y Axis", {}))
            plot.update_yaxes(**{
                "ticks": "outside",
                "gridcolor": "#1d2021",
                #"font": {"size": 16},
                "showgrid": True,
                "zeroline": True,
                "showline": True,
                "zerolinecolor": "#1d2021",
            })
        elif isinstance(metric_columns, list):
            plot_type = tag_dict.get(data.get("Plot Type", "Box Plot"))
            normalise_by = data.get("Normalise By", "None")
            plots = {
                metric: results_plot(plot_type, grouped, metric, normalise_by)
                for metric in metric_columns
            }
            plot_type = 'table' if plot_type == "results-table-tab" else "xy"
            plot = make_subplots(
                cols=1,
                rows=len(plots),
                vertical_spacing=0.01,
                shared_xaxes=True,
                specs = [[{'type': plot_type}] for _ in range(len(plots))]
            )
            for index, (metric, subplot) in enumerate(plots.items(), start=1):
                axes_num = "" if index == 1 else str(index)
                traces = subplot.select_traces()
                plot_config = subplot.to_dict()
                layout = plot_config.get("layout", {})
                print(plot_config["layout"].keys())
                for trace in traces:
                    plot.add_trace(
                        trace,
                        row=index,
                        col=1
                    )
                if plot_type != 'table':
                    plot.add_annotation(
                        xref='x domain',
                        yref='y domain',
                        x=0.01,
                        y=0.97,
                        text=f'({chr(96+index)})',
                        font={"size": 24},
                        showarrow=False,
                        row=index,
                        col=1
                    )
                plot.update_layout(**{
                    k: v
                    for k, v in layout.items()
                    if k not in ["xaxis", "yaxis"]
                })
                xaxis_config = layout.get('xaxis', {})
                xaxis_config.update(data.get("X Axis", {}).get(metric, {}))
                xaxis_config.update({
                    "ticks": "outside",
                    "gridcolor": "#b6bdbf",
                    "showgrid": True,
                    "showline": True,
                    "mirror": True,
                    "linecolor": "#1d2021",
                    #"title": {"font": {"size": 16}},
                    "tickangle": -45
                })
                yaxis_config = layout.get('yaxis', {})
                yaxis_config.update(data.get("Y Axis", {}).get(metric, {}))
                yaxis_config.update({
                    "ticks": "outside",
                    "gridcolor": "#b6bdbf",
                    "showgrid": True,
                    "zeroline": True,
                    "showline": True,
                    #"title": {"font": {"size": 16}},
                    "mirror": True,
                    "zerolinecolor": "#b6bdbf",
                    "linecolor": "#1d2021",
                    "minor": {
                        "ticks": "outside",
                        "gridcolor": "#e5e6e7",
                        "showgrid": True,

                    }
                })
                if plot_type != 'table':
                    plot.layout[f'xaxis{axes_num}'].update(**xaxis_config)
                    plot.layout[f'yaxis{axes_num}'].update(**yaxis_config)
        else:
            plot = results_plot(
                tag_dict.get(data.get("Plot Type", "Box Plot")),
                grouped,
                "r2",
                data.get("Normalise By", "None"),
            )
            plot.update_xaxes(**data.get("X Axis", {}))
            plot.update_yaxes(**data.get("Y Axis", {}))
            plot.update_yaxes(**{
                "ticks": "outside",
                "gridcolor": "#1d2021",
                "showgrid": True,
                "zeroline": True,
                "zerolinecolor": "#1d2021",
            })
        dimension_dict = {
            "width": 2100,
            "height": 800 * (
                len(metric_columns) if isinstance(metric_columns, list) else 1
            )
        }
        layout_dict = data.get("Layout", {})
        plot.update_layout(
            **layout_dict,
            **{k: v for k, v in dimension_dict.items() if k not in layout_dict}
        )
        plot.write_image(filepath)


@callback(
    Output("graph-config-json", "data"),
    Input("download-config", "n_clicks"),
    State("reference-options", "value"),
    State("field-options", "value"),
    State("calibrated-device-options", "value"),
    State("technique-options", "value"),
    State("scaling-options", "value"),
    State("var-options", "value"),
    State("fold-options", "value"),
    State("result-box-plot-split-by", "value"),
    State("norm-options", "value"),
    State("remove-outliers", "value"),
    State("graph-type", "value"),
    State("metric-options", "value"),
    prevent_initial_call=True,
)
def download_config(
    _,
    ref,
    field,
    cal,
    tech,
    scal,
    var,
    fold,
    splitby,
    norm,
    outliers,
    graph_type,
    metric,
):
    """ """
    tag_dict = {
        "results-table-tab": "Results Table",
        "box-plot-tab": "Box Plot",
        "violin-plot-tab": "Violin Plot",
        "prop-imp-plot-tab": "Proportion Improved",
        "target-plot-tab": "Target Diagram",
    }
    config = {
        "Reference Instruments": ref,
        "Fields": field,
        "Calibration Devices": cal,
        "Techniques": tech,
        "Scaling Techniques": scal,
        "Variables": var,
        "Folds": fold,
        "Split By": splitby,
        "Normalise By": norm,
        "Remove Outliers": outliers,
        "Plot Type": tag_dict.get(graph_type),
        "Metric Column": metric,
    }
    config_json = json.dumps(
        {key: val for key, val in config.items() if val is not None}, indent=4
    )
    return {"content": config_json, "filename": "config.json"}


item_stores = [
    dcc.Store(id="folder-path"),
    dcc.Store(id="reference-df-index"),
    dcc.Store(id="db-index"),
    dcc.Store(id="results-df"),
    dcc.Store(id="raw-df"),
    dcc.Store(id="chosen-combo-index"),
    dcc.Store(id="split-data"),
    dcc.Store(id="graphs-generated-text"),
    dcc.Store(id="ref-colours"),
]

top_row = [
    dbc.Row(
        [html.Div("Graddnodi", className="h1", style={"text-align": "center"})]
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Dropdown(
                        sorted(output_options),
                        #sorted(output_options)[0],
                        id="folder-name",
                        multi=True,
                        placeholder="Select comparisons"
                    ),
                ]
            ),
            dbc.Col(
                [
                    dcc.Dropdown(
                        id="reference-options",
                        multi=True,
                        placeholder="Reference Devices"
                    ),
                ]
            ),
            dbc.Col(
                [
                    html.Button(
                        id="reference-button-state",
                        n_clicks=0,
                        children="1) Query Configuration Options"
                    ),
                ]
            )
        ]
    ),
]

checklist_options = {"overflow-y": "scroll", "height": "20vh"}

selections = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H4("Fields"),
                    dcc.Checklist(id="field-options", style=checklist_options),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Calibration Candidates"),
                    dcc.Checklist(
                        id="calibrated-device-options", style=checklist_options
                    ),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Regression Techniques"),
                    dcc.Checklist(
                        id="technique-options", style=checklist_options
                    ),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Scaling Techniques"),
                    dcc.Checklist(
                        id="scaling-options", style=checklist_options
                    ),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Variable Combos"),
                    dcc.Checklist(id="var-options", style=checklist_options),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Folds"),
                    dcc.Checklist(id="fold-options", style=checklist_options),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Normalise Data"),
                    dcc.RadioItems(
                        id="norm-options",
                        style=checklist_options
                    ),
                ],
            ),
            dbc.Col(
                [
                    html.H4("Remove Outliers"),
                    dcc.RadioItems(["Yes", "No"], "Yes", id="remove-outliers"),
                ],
            ),
            dbc.Col(
                [
                    html.H4("Split By"),
                    dcc.Checklist(
                        [
                            "Reference",
                            "Calibrated",
                            "Field",
                            "Technique",
                            "Scaling Method",
                            "Variables",
                            "Fold",
                        ],
                        ["Calibrated"],
                        id="split-by-options",
                        style=checklist_options
                    ),
                ],
            ),
            dbc.Col(
                [
                    html.H4("Metric"),
                    dcc.RadioItems(
                        id="metric-options",
                        value="r2",
                        style=checklist_options
                    ),
                ],
            ),
        ]
    )
]

results_table = [
    dbc.Row(
        [html.Div("Results", className="h2", style={"text-align": "center"})]
    ),
    dbc.Row(
        [dcc.Graph(figure={}, id="results-table")],
    ),
]


graph_selection_row = [
    dbc.Row(
        [
        ]
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Button(
                        id="submit-button-state", n_clicks=0, children="Submit"
                    ),
                ],
                width=2,
            ),
            dbc.Col(
                html.Div(id="num-of-runs", style={"text-align": "center"}),
                width=8,
            ),
            dbc.Col(
                [
                    html.Button("Download config", id="download-config"),
                    dcc.Download(id="graph-config-json"),
                ],
                width=2,
            ),
        ]
    ),
    dcc.Tabs(
        id="graph-type",
        value="results-table-tab",
        children=[
            dcc.Tab(label="Results Table", value="results-table-tab"),
            dcc.Tab(label="Box Plot", value="box-plot-tab"),
        ],
    ),
]

summary_graph = [dcc.Graph(figure={}, id="summary-figure")]

generate_graphs = [
    dcc.Upload(id='generate-graphs-button', children=html.Button("Upload Batch Configuration")),
]

app.layout = dbc.Container(
    [
        *item_stores,
        *top_row,
        html.Hr(),
        *selections,
        html.Hr(),
        *graph_selection_row,
        html.Hr(),
        *summary_graph,
        html.Hr(),
        *generate_graphs,
        html.Hr(),
        # *results_table,
        # html.Hr(),
        # *results_box_plots,
    ],
    fluid=True
)


if __name__ == "__main__":
    app.run(debug=True)
