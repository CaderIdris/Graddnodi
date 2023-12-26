from functools import partial
from io import StringIO
import logging
import os
from pathlib import Path
import sqlite3 as sql

from dash import Dash, html, dash_table, dcc, callback, Output, Input
from flask import Flask
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

FULL_TABLE_LAYOUT = {"margin": {"pad": 0, "b": 0, "t": 0, "l": 0, "r": 0}}

external_stylesheets = [dbc.themes.JOURNAL]

flask_server = Flask(__name__)
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

output_folder = Path(os.getenv("GRADDNODI_OUTPUT", "Output/"))
output_options = [
    str(dir.parts[-1]) for dir in output_folder.glob("*") if dir.is_dir()
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
        sql='SELECT DISTINCT "Reference", "Field", "Calibrated", "Technique", "Scaling Method", "Variables" FROM Results;',
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
    Output("chosen-combo-index", "data"),
    Output("num-of-runs", "children"),
    Input("db-index", "data"),
    Input("reference-options", "value"),
    Input("field-options", "value"),
    Input("calibrated-device-options", "value"),
    Input("technique-options", "value"),
    Input("scaling-options", "value"),
    Input("var-options", "value"),
)
def filter_options(data, ref_d, fields, cal_d, tech, sca, var):
    levels = {
        "Field": fields,
        "Calibrated": cal_d,
        "Technique": tech,
        "Scaling Method": sca,
        "Variables": var,
    }
    db_index = pd.read_json(StringIO(data), orient="split")
    df = db_index[db_index["Reference"] == ref_d]
    s_df = df.copy(deep=True)
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
        s_df.to_json(orient="split"),
        # table_fig,
        f"{s_df.shape[0]} combinations",
    )


@callback(
    Output("results-df", "data"),
    Output("results-table", "figure"),
    Input("chosen-combo-index", "data"),
    Input("folder-path", "data"),
    Input("reference-options", "value"),
)
def get_results_df(data, path, ref_d):
    if not ref_d:
        return (
            pd.DataFrame().to_json(orient="split"),
            go.Figure(data=[go.Table(header={}, cells={})]),
        )
    df = pd.read_json(StringIO(data), orient="split")
    query_list = ["SELECT *", "FROM Results"]
    for i, (name, vals) in enumerate(df.items()):
        val_list = "', '".join(vals.unique())
        query_list.append(
            f"""{"WHERE" if i == 0 else "AND"} "{name}" in ('{val_list}')"""
        )
    con = sql.connect(Path(path).joinpath("Results").joinpath("Results.db"))
    query = "\n".join(query_list)
    sql_data = pd.read_sql(sql=f"{query};", con=con).drop(
        columns=['level_0', 'index'],
        errors='ignore'
    )
    con.close()

    table_cell_format = [
        ".2f" if i == "float64" else "" for i in sql_data.dtypes
    ]

    table_fig = go.Figure(
        data=[
            go.Table(
                header={"values": list(sql_data.columns), "align": "left"},
                cells={
                    "values": sql_data.transpose().values.tolist(),
                    "align": "left",
                    "format": table_cell_format,
                },
            )
        ],
        layout=FULL_TABLE_LAYOUT,
    )

    return sql_data.to_json(orient="split"), table_fig


@callback(
    Output("result-box-plot-fig", "figure"),
    Output("result-box-plot-table", "figure"),
    Input("results-df", "data"),
    Input("result-box-plot-tabs", "value"),
    Input("result-box-plot-split-by", "value"),
)
def results_box_plot(data, tab, splitby):
    """ """
    df = pd.read_json(StringIO(data), orient="split")
    box_plot_fig = go.Figure()
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
        df = df.loc[:, [*index, tab]]
        grouped = df.groupby(splitby)
    except KeyError:
        return (
            go.Figure(data=[go.Table(header={}, cells={})]),
            go.Figure(data=[go.Table(header={}, cells={})]),
        )

    summary = pd.DataFrame()

    summary_functions = {
        "Count": len,
        "Mean": np.mean,
        "Min": np.min,
        "25%": partial(np.quantile, q=0.25),
        "50%": partial(np.quantile, q=0.5),
        "75%": partial(np.quantile, q=0.75),
        "Max": np.max,
    }
    lower_bounds = list()
    upper_bounds = list()

    for name, data in grouped:
        q3 = data[tab].quantile(0.75) 
        q1 = data[tab].quantile(0.25)
        iqr = q3 - q1
        upper_bounds.append(q3 + (2 * iqr))
        lower_bounds.append(q1 - (2 * iqr))
        violin_data = data[tab]
        violin_data = violin_data[violin_data.ge(q1 - (2 * iqr)) & violin_data.le(q3 + (2 * iqr))]
        box_plot_fig.add_trace(
#            go.Box(
#                y=data[tab].values,
#                name=name,
#                boxpoints="all",
#                hoverinfo="all",
#                text=data.loc[:, index]
#                .to_csv(header=False, index=False)
#                .split("\n"),
#            )
            go.Violin(
                y=violin_data,
                name=name,
                box_visible=True,
                hoverinfo="name",
                spanmode="manual",
                span=(q1 - (2 * iqr), q3 + (2 * iqr)),
                meanline_visible=True
            )
        )
        for stat, func in summary_functions.items():
            summary.loc[stat, name] = float(func(data[tab].values))

    box_plot_fig.update_layout(
        yaxis={
            "autorange": False,
            "range": (min(lower_bounds), max(upper_bounds))
        },
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

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

    return box_plot_fig, table_fig


item_stores = [
    dcc.Store(id="folder-path"),
    dcc.Store(id="db-index"),
    dcc.Store(id="results-df"),
    dcc.Store(id="chosen-combo-index"),
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
                    dcc.RadioItems(
                        id="reference-options", style=checklist_options
                    )
                ]
            ),
            dbc.Col(
                [dcc.Checklist(id="field-options", style=checklist_options)]
            ),
            dbc.Col(
                [
                    dcc.Checklist(
                        id="calibrated-device-options", style=checklist_options
                    )
                ]
            ),
            dbc.Col(
                [dcc.Checklist(id="technique-options", style=checklist_options)]
            ),
            dbc.Col(
                [dcc.Checklist(id="scaling-options", style=checklist_options)]
            ),
            dbc.Col([dcc.Checklist(id="var-options", style=checklist_options)]),
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
    dbc.Row([html.Div(id="num-of-runs", style={"text-align": "center"})]),
]

results_box_plots = [
    dcc.Tabs(
        id="result-box-plot-tabs",
        value="r2",
        children=[
            dcc.Tab(
                label="Explained Variance Score",
                value="Explained Variance Score",
            ),
            dcc.Tab(label="Max Error", value="Max Error"),
            dcc.Tab(label="Mean Absolute Error", value="Mean Absolute Error"),
            dcc.Tab(
                label="Root Mean Squared Error", value="Root Mean Squared Error"
            ),
            dcc.Tab(
                label="Median Absolute Error", value="Median Absolute Error"
            ),
            dcc.Tab(
                label="Mean Absolute Percentage Error",
                value="Mean Absolute Percentage Error",
            ),
            dcc.Tab(label="r2", value="r2"),
        ],
    ),
    dbc.Col(
        dcc.RadioItems(
            [
                "Calibrated",
                "Field",
                "Technique",
                "Scaling Method",
                "Variables",
                "Fold",
            ],
            "Calibrated",
            id="result-box-plot-split-by",
        ),
        width=2,
    ),
    dbc.Col(
        [
            dcc.Graph(figure={}, id="result-box-plot-fig"),
            dcc.Graph(figure={}, id="result-box-plot-table"),
        ]
    ),
]


app.layout = dbc.Container(
    [
        *item_stores,
        *top_row,
        html.Hr(),
        *selections,
        html.Hr(),
        *results_table,
        html.Hr(),
        *results_box_plots,
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
