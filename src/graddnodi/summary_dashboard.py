from functools import partial
import json
from io import StringIO
import logging
import os
from pathlib import Path
import sqlite3 as sql

from dash import Dash, html, dcc, callback, Output, Input, State
from flask import Flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

FULL_TABLE_LAYOUT = {"margin": {"pad": 0, "b": 0, "t": 0, "l": 0, "r": 0}}

FIGURE_LAYOUT = {
    "margin": {"pad": 0, "b": 0, "t": 0, "l": 0, "r": 0},
    "showlegend": False,
    "font": {"size": 16},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "colorway": ["#1d2021"],
    "scattermode": "group",
}

external_stylesheets = [dbc.themes.JOURNAL]

flask_server = Flask(__name__)
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    server=flask_server,
)
server = app.server

output_folder = Path(os.getenv("GRADDNODI_OUTPUT", "Output/"))
logging.critical(output_folder)
logging.critical(output_folder.is_dir())
output_options = [
    str(_dir.parts[-1]) for _dir in output_folder.glob("*") if _dir.is_dir()
]


@callback(Output("folder-path", "data"), Input("folder-name", "value"))
def folder_path(name):
    return str(output_folder.joinpath(name))


@callback(
    Output("folder-output-text", "children"), Input("folder-path", "data")
)
def folder_path_text(path):
    return f"Using data from {path}"


@callback(Output("db-index", "data"), Input("folder-path", "data"))
def get_df_index(path):
    con = sql.connect(Path(path).joinpath("Results").joinpath("Results.db"))
    raw_index = pd.read_sql(
        sql='SELECT DISTINCT "Reference", "Field", "Calibrated", "Technique", "Scaling Method", "Variables", "Fold" FROM Results;',
        con=con,
    )
    con.close()
    return raw_index.to_json(orient="split")


@callback(Output("reference-options", "options"), Input("db-index", "data"))
def ref_opts(data):
    df = pd.read_json(StringIO(data), orient="split")
    return [{"label": i, "value": i} for i in sorted(df["Reference"].unique())]


@callback(
    Output("field-options", "options"),
    Output("calibrated-device-options", "options"),
    Output("technique-options", "options"),
    Output("scaling-options", "options"),
    Output("var-options", "options"),
    Output("fold-options", "options"),
    Output("chosen-combo-index", "data"),
    Output("num-of-runs", "children"),
    Input("submit-button-state", "n_clicks"),
    Input("db-index", "data"),
    Input("reference-options", "value"),
    State("field-options", "value"),
    State("calibrated-device-options", "value"),
    State("technique-options", "value"),
    State("scaling-options", "value"),
    State("var-options", "value"),
    State("fold-options", "value"),
)
def filter_options(_, data, ref_d, fields, cal_d, tech, sca, var, fold):
    levels = {
        "Field": fields,
        "Calibrated": cal_d,
        "Technique": tech,
        "Scaling Method": sca,
        "Variables": var,
        "Fold": fold,
    }
    if not ref_d:
        return [], [], [], [], [], [], "", ""
    if not isinstance(data, pd.DataFrame):
        db_index = pd.read_json(StringIO(data), orient="split")
    else:
        db_index = data
    df: pd.DataFrame = db_index[db_index["Reference"].isin(ref_d)]
    s_df: pd.DataFrame = df.copy(deep=True)
    for name, col in levels.items():
        if not col:
            cols = s_df[name].unique()
        else:
            cols = col
        s_df = s_df[s_df[name].isin(cols)]

    return (
        [{"label": i, "value": i} for i in sorted(df["Field"].unique())],
        [{"label": i, "value": i} for i in sorted(df["Calibrated"].unique())],
        [
            {"label": i.replace(" Regression", ""), "value": i}
            for i in sorted(df["Technique"].unique())
        ],
        [
            {"label": i, "value": i}
            for i in sorted(df["Scaling Method"].unique())
        ],
        [{"label": i, "value": i} for i in sorted(df["Variables"].unique())],
        [{"label": i, "value": i} for i in sorted(df["Fold"].unique())],
        s_df.to_json(orient="split"),
        # table_fig,
        f"{s_df.shape[0]} combinations",
    )


@callback(
    Output("results-df", "data"),
    Output("raw-df", "data"),
    # Output("results-table", "figure"),
    Input("chosen-combo-index", "data"),
    State("folder-path", "data"),
    State("reference-options", "value"),
)
def get_results_df(data, path, ref_d):
    if not ref_d:
        return (
            pd.DataFrame().to_json(orient="split"),
            pd.DataFrame().to_json(orient="split"),
        )
    df = pd.read_json(StringIO(data), orient="split")
    query_list = ["SELECT *", "FROM Results"]
    none_query_list = ["SELECT *", "FROM Results"]
    for i, (name, vals) in enumerate(df.items()):
        val_list = "', '".join(vals.unique())
        query_list.append(
            f"""{"WHERE" if i == 0 else "AND"} "{name}" in ('{val_list}')"""
        )
        if name == "Technique":
            none_query_list.append(
                f"""{"WHERE" if i == 0 else "AND"} "{name}" in ('None')"""
            )
        else:
            none_query_list.append(
                f"""{"WHERE" if i == 0 else "AND"} "{name}" in ('{val_list}')"""
            )
    con = sql.connect(Path(path).joinpath("Results").joinpath("Results.db"))
    query = "\n".join(query_list)
    none_query = "\n".join(none_query_list)
    sql_data = pd.read_sql(sql=f"{query};", con=con).drop(
        columns=["level_0", "index"], errors="ignore"
    )
    none_sql_data = pd.read_sql(sql=f"{none_query};", con=con).drop(
        columns=["level_0", "index"], errors="ignore"
    )
    con.close()

    return sql_data.to_json(orient="split"), none_sql_data.to_json(
        orient="split"
    )  # , table_fig


@callback(Output("results-tabs", "children"), Input("folder-path", "data"))
def results_tabs(path):
    """ """
    con = sql.connect(Path(path).joinpath("Results").joinpath("Results.db"))
    df = pd.read_sql("SELECT * FROM 'Results' LIMIT 1;", con=con)
    con.close()
    bad_cols = [
        "index",
        "Field",
        "Reference",
        "Calibrated",
        "Technique",
        "Scaling Method",
        "Variables",
        "Fold",
        "Count",
        *[col for col in df.columns if "Reference" in col],
    ]
    cols = [col for col in df.columns if col not in bad_cols]
    tabs = list()
    for col in cols:
        tabs.append(dcc.Tab(label=col, value=col))

    if not tabs:
        tabs = [dcc.Tab(label="No Data", value="nd")]

    return dcc.Tabs(
        id="result-box-plot-tabs",
        value=cols[0] if cols else "nd",
        children=tabs,
    )


def full_results_table_plot(df):
    """ """
    df = df.drop(
        columns=[
            col for col in df.columns if "(Raw)" in col or "Reference " in col
        ]
    )
    table_cell_format = [".2f" if i == "float64" else "" for i in df.dtypes]
    return go.Figure(
        data=go.Table(
            header={"values": list(df.columns)},
            cells={
                "values": df.transpose().values.tolist(),
                "format": table_cell_format,
            },
        ),
        layout=FULL_TABLE_LAYOUT,
    )


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


@callback(
    Output("split-data", "data"),
    Input("results-df", "data"),
    State("result-box-plot-split-by", "value"),
    State("norm-options", "value"),
    State("remove-outliers", "value"),
    State("raw-df", "data"),
)
def grouped_data(
    cal_results, split_by, norm_by, outlier_removal, uncal_results
):
    """ """
    s_data = dict()
    # Import data
    df = pd.read_json(StringIO(cal_results), orient="split")
    if not df.shape[0]:
        return s_data
    raw_df = pd.read_json(StringIO(uncal_results), orient="split")
    df = df.join(
        raw_df.set_index(["Field", "Reference", "Calibrated"]),
        on=["Field", "Reference", "Calibrated"],
        how="left",
        rsuffix=" (Raw)",
    )

    # config
    no_norm = [
        "Mean Absolute Percentage Error",
        "r2",
        "Explained Variance Score",
    ]

    index = [
        "Field",
        "Reference",
        "Calibrated",
        "Technique",
        "Scaling Method",
        "Variables",
        "Fold",
    ]

    try:
        grouped = df.groupby(split_by)
    except KeyError:
        return s_data

    logging.critical(outlier_removal)
    logging.critical(norm_by)

    for name_t, data in grouped:
        name = ", ".join(name_t)
        for col in [
            col
            for col in data.columns
            if "(Raw)" not in col
            and "Reference " not in col
            and col not in index
        ]:
            if outlier_removal == "Yes":
                q3 = data[col].quantile(0.75)
                q1 = data[col].quantile(0.25)
                iqr = q3 - q1
                upper_bound = q3 + (2 * iqr)
                lower_bound = q1 - (2 * iqr)
                no_outliers_col = data[col].copy()
                no_outliers_col[
                    np.logical_or(
                        no_outliers_col.gt(upper_bound),
                        no_outliers_col.lt(lower_bound),
                    )
                ] = np.nan
                data[col] = no_outliers_col
            if (
                norm_by is not None
                and norm_by != "None"
                and col not in no_norm
            ):
                data[col] = data[col].div(data[norm_by])
                data[f"{col} (Raw)"] = data[f"{col} (Raw)"].div(data[norm_by])
        if "Technique" not in split_by:
            data = data[data["Technique"] != "None"]
        s_data[name] = data.to_json(orient="split")
    return s_data


def result_table_plot(dfs, col):
    """ """
    summary_functions = {
        "Count": len,
        "Mean": np.mean,
        "σ": np.std,
        "Min": np.min,
        "25%": partial(np.nanquantile, q=0.25),
        "50%": partial(np.nanquantile, q=0.5),
        "75%": partial(np.nanquantile, q=0.75),
        "Max": np.max,
    }
    summary = pd.DataFrame()
    for name, df in dfs.items():
        c_data = df[col]
        for stat, func in summary_functions.items():
            summary.loc[stat, name] = round(func(c_data), 3)
    summary.index.name = "Statistic"
    summary = summary.reset_index()
    summary.insert(0, "Statistic", summary.pop("Statistic"))
    table_fig = go.Figure(
        data=[
            go.Table(
                header={"values": list(summary.columns), "align": "left"},
                cells={
                    "values": summary.transpose().values.tolist(),
                    "align": "left",
                },
            )
        ],
        layout=FULL_TABLE_LAYOUT,
    )
    return table_fig


def box_plot(dfs, col, norm_options):
    """ """
    no_norm = [
        "Mean Absolute Percentage Error",
        "r2",
        "Explained Variance Score",
    ]
    box_plot_fig = go.Figure()
    for name, df in dfs.items():
        c_data = df[col]
        box_plot_fig.add_trace(
            go.Box(
                y=c_data,
                name=name,
                hoverinfo="all",
                boxmean="sd",
                fillcolor="rgba(0,0,0,0)",
            )
        )

    if norm_options and norm_options != "None" and col not in no_norm:
        title = f"{col} / {norm_options}"
    else:
        title = col

    box_plot_fig.update_layout(
        yaxis={"autorange": False, "title": title},
        **FIGURE_LAYOUT,
        width=1200,
        height=600,
    )
    return box_plot_fig


def violin_plot(dfs, col, norm_options):
    """ """
    no_norm = [
        "Mean Absolute Percentage Error",
        "r2",
        "Explained Variance Score",
    ]
    box_plot_fig = go.Figure()
    for name, df in dfs.items():
        c_data = df[col]
        box_plot_fig.add_trace(
            go.Violin(
                y=c_data,
                name=name,
                box_visible=True,
                hoverinfo="name",
                meanline_visible=True,
                fillcolor="rgba(0,0,0,0)",
            )
        )

    if norm_options and norm_options != "None" and col not in no_norm:
        title = f"{col} / {norm_options}"
    else:
        title = col

    box_plot_fig.update_layout(
        yaxis={"autorange": False, "title": title},
        **FIGURE_LAYOUT,
        width=1500,
        height=750,
    )
    return box_plot_fig


def prop_imp_plot(dfs, col):
    """ """
    pos_better = ["r2", "Explained Variance Score"]
    improved_techs = pd.DataFrame()
    for name, df in dfs.items():
        adjusted = df[col] - df[f"{col} (Raw)"]
        if col not in pos_better:
            improved = adjusted.lt(0)
        else:
            improved = adjusted.gt(0)
        improved_techs.loc["Improved", name] = improved.sum() / improved.size
        improved_techs.loc["Worsened", name] = (
            improved.size - improved.sum()
        ) / improved.size
    bars = [
        go.Bar(
            name=name,
            x=data.index,
            y=data.values,
            marker_line_color="rgba(0,0,0,0)",
            marker_color="#1FE41B" if name == "Improved" else "#E41B1F",
        )
        for name, data in improved_techs.iterrows()
    ]

    improved_bar = go.Figure(
        data=bars,
        layout={
            "width": 1500,
            "height": 750,
            **FIGURE_LAYOUT,
        },
    )
    return improved_bar


def target_plot(dfs, norm_options):
    """ """
    fig = go.Figure()
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1)

    ind_cols = [
        "Field",
        "Reference",
        "Calibrated",
        "Technique",
        "Scaling Method",
        "Variables",
        "Fold",
    ]

    for name, df in dfs.items():
        df["text"] = df[ind_cols[0]]
        for col in ind_cols[1:]:
            df["text"] = df["text"].str.cat(df[col].astype(str), sep=", ")
        if norm_options and norm_options not in ["None"]:
            crmse = df["Centered Root Mean Squared Error"].mul(norm_options)
            bias = df["Mean Bias Error"].mul(norm_options)
        else:
            crmse = df["Centered Root Mean Squared Error"]
            bias = df["Mean Bias Error"]
        crmse = crmse.div(df["Reference Standard Deviation"])
        bias = bias.div(df["Reference Standard Deviation"])
        fig.add_trace(
            go.Scatter(
                x=crmse, y=bias, name=name, text=df["text"], mode="markers"
            )
        )
    fig.update_xaxes(
        range=[-2, 6],
        minallowed=-2,
        maxallowed=6,
    )

    fig.update_yaxes(
        range=[-2, 6],
        minallowed=-2,
        maxallowed=6,
    )
    fig.update_layout(**FIGURE_LAYOUT)

    fig.update_layout(
        margin={"autoexpand": False, "b": 75, "t": 75, "l": 75, "r": 575},
        legend={"itemclick": "toggleothers", "itemdoubleclick": "toggle"},
        xaxis_title=r"cRMSE / Reference σ",
        yaxis_title=r"Bias / Reference σ",
        autosize=False,
        width=1500,
        height=1000,
        minreducedwidth=950,
        minreducedheight=950,
    )

    return fig


@callback(
    Output("summary-figure", "figure"),
    Input("graph-type", "value"),
    Input("split-data", "data"),
    Input("result-box-plot-tabs", "value"),
    State("norm-options", "value"),
)
def results_plot(plot_type, dfs, col, norm):
    logging.critical(f"col: {col}")
    if dfs:
        data = {
            key: pd.read_json(StringIO(df), orient="split")
            for key, df in dfs.items()
        }
    else:
        return empty_figure()

    if plot_type == "results-table-tab":
        return result_table_plot(data, col)
    elif plot_type == "box-plot-tab":
        return box_plot(data, col, norm)
    elif plot_type == "violin-plot-tab":
        return violin_plot(data, col, norm)
    elif plot_type == "prop-imp-plot-tab":
        return prop_imp_plot(data, col)
    elif plot_type == "target-plot-tab":
        return target_plot(data, norm)
    elif plot_type == "valerio-temp":
        return empty_figure(plot_type)


@callback(Output("norm-options", "options"), Input("results-df", "data"))
def norm_options(data):
    if not isinstance(data, pd.DataFrame):
        df = pd.read_json(StringIO(data), orient="split")
    else:
        df = data
    valid_options = [
        "Reference Mean",
        "Reference Standard Deviation",
        "Reference Interquartile Range",
        "Reference Range",
    ]
    return [
        {"label": "None", "value": "None", "disabled": False},
        *[
            {
                "label": i.replace("Reference ", ""),
                "value": i,
                "disabled": i not in df.columns,
            }
            for i in valid_options
        ],
    ]


@callback(
    Output("graphs-generated-text", "data"),
    Input("generate-graphs-button", "n_clicks"),
    State("folder-path", "data"),
    State("db-index", "data"),
    prevent_initial_call=True,
)
def generate_graphs_for_paper(button, path, dbi):
    tag_dict = {
        "Results Table": "results-table-tab",
        "Box Plot": "box-plot-tab",
        "Violin Plot": "violin-plot-tab",
        "Proportion Improved": "prop-imp-plot-tab",
        "Target Diagram": "target-plot-tab",
    }
    graph_path = Path(path).joinpath("Graphs")
    graph_path.mkdir(parents=True, exist_ok=True)
    if button == 0:
        return ""
    if not isinstance(dbi, pd.DataFrame):
        db_index = pd.read_json(StringIO(dbi), orient="split")
    else:
        db_index = dbi
    with Path(path).joinpath("Graphs/summary_graphs.json").open("r") as file:
        config = json.load(file)
    logging.error(config)
    for run, data in config.items():
        logging.error(run)
        filepath = graph_path.joinpath(f"{run}.svg")
        if filepath.exists():
            continue
        _, _, _, _, _, _, cci, _ = filter_options(
            None,
            db_index,
            data.get("Reference Instruments", list()),
            data.get("Fields", list()),
            data.get("Calibration Devices", list()),
            data.get("Techniques", list()),
            data.get("Scaling Techniques", list()),
            data.get("Variables", list()),
            data.get("Folds", list()),
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
        logging.error(grouped.keys())
        plot = results_plot(
            tag_dict.get(data.get("Plot Type", "Box Plot")),
            grouped,
            data.get("Metric Column", "r2"),
            data.get("Normalise By", "None"),
        )
        plot.update_xaxes(**data.get("X Axis", {}))
        plot.update_yaxes(**data.get("Y Axis", {}))
        plot.update_layout(**data.get("Layout", {}))
        logging.critical(data.get("Y Axis", {}))
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
    State("result-box-plot-tabs", "value"),
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
    dcc.Store(id="db-index"),
    dcc.Store(id="results-df"),
    dcc.Store(id="raw-df"),
    dcc.Store(id="chosen-combo-index"),
    dcc.Store(id="split-data"),
    dcc.Store(id="graphs-generated-text"),
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
                        sorted(output_options)[0],
                        id="folder-name",
                    ),
                ]
            ),
            dbc.Col([html.Div(id="folder-output-text")]),
        ]
    ),
]

checklist_options = {"overflow-y": "scroll", "height": "20vh"}

selections = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H4("Reference Devices"),
                    dcc.Checklist(
                        id="reference-options", style=checklist_options
                    ),
                ]
            ),
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
                        id="result-box-plot-split-by",
                    ),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    html.H4("Normalise Data"),
                    dcc.RadioItems(
                        id="norm-options",
                    ),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    html.H4("Remove Outliers"),
                    dcc.RadioItems(["Yes", "No"], "Yes", id="remove-outliers"),
                ],
                width=2,
            ),
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
            dcc.Tab(label="Violin Plot", value="violin-plot-tab"),
            dcc.Tab(label="Proportion Improved", value="prop-imp-plot-tab"),
            dcc.Tab(label="Target Plot", value="target-plot-tab"),
            dcc.Tab(label="The Plot Valerio Suggested", value="valerio-temp"),
        ],
    ),
    html.Div(id="results-tabs"),
]

summary_graph = [dcc.Graph(id="summary-figure")]

generate_graphs = [
    html.Button(id="generate-graphs-button", n_clicks=0, children="Generate"),
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
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
