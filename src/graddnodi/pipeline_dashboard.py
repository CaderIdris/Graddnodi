from functools import partial
import json
from io import StringIO
import logging
import os
from pathlib import Path
import pickle
import sqlite3 as sql
from typing import Optional, TYPE_CHECKING

from dash import Dash, html, dcc, callback, Output, Input, State
from flask import Flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc


FULL_TABLE_LAYOUT = {"margin": {"pad": 0, "b": 0, "t": 0, "l": 0, "r": 0}}

FIGURE_LAYOUT = {
    "margin": {"pad": 0, "b": 40, "t": 40, "l": 40, "r": 0},
    "showlegend": False,
    "font": {"size": 10},
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
output_options = [
    str(_dir.parts[-1]) for _dir in output_folder.glob("*") if _dir.is_dir()
]


@callback(
    Output("folder-path", "data"),
    Input("folder-name", "value"),
)
def folder_path(name: str) -> str:
    """Return path to folder containing results to analyse."""
    return output_folder.joinpath(name).as_posix()


@callback(
    Output("folder-output-text", "children"),
    Input("folder-path", "data"),
)
def folder_path_text(path: str) -> str:
    """Output string representation of path."""
    return f"Using data from {path}"

def get_glob(path: Path) -> list[dict[str, str]]:
    """Return directories at a specified path."""
    return [
        {"label": i.parts[-1], "value": i.as_posix()}
        for i in sorted(path.glob("*")) if i.is_dir()
    ]


@callback(
    Output("comparison-options", "options"),
    Input("folder-path", "data"),
    prevent_initial_call=True
)
def comparison_opts(path_str: str) -> list[dict[str, str]]:
    """Get all comparison directories."""
    path = Path(path_str).joinpath('Pipelines')
    return get_glob(path)


@callback(
    Output("field-options", "options"),
    Input("comparison-options", "value"),
    prevent_initial_call=True
)
def field_opts(path_str: str) -> list[dict[str, str]]:
    """Get all field directories."""
    path = Path(path_str)
    return get_glob(path)


@callback(
    Output("technique-options", "options"),
    Input("field-options", "value"),
    prevent_initial_call=True
)
def tech_opts(path_str: str) -> list[dict[str, str]]:
    """Get all technique directories."""
    path = Path(path_str)
    return get_glob(path)


@callback(
    Output("scaling-options", "options"),
    Input("technique-options", "value"),
    prevent_initial_call=True
)
def scaling_opts(path_str: str) -> list[dict[str, str]]:
    """Get all scaling directories."""
    path = Path(path_str)
    return get_glob(path)


@callback(
    Output("variable-options", "options"),
    Input("scaling-options", "value"),
    prevent_initial_call=True
)
def variable_opts(path_str: str) -> list[dict[str, str]]:
    """Get all variable directories."""
    path = Path(path_str)
    return get_glob(path)


@callback(
    Output("x-measurements", "data"),
    Output("y-measurements", "data"),
    Input("run-button", "n_clicks"),
    State("folder-path", "data"),
    State("comparison-options", "value"),
    State("field-options", "value"),
    prevent_initial_call=True
)
def get_data(
        _: int,
        folder_path: str,
        comparison_str: str,
        field_str: str
    ) -> tuple[str, str]:
    """Get comparison measurements."""
    comparison = Path(comparison_str).parts[-1]
    field = Path(field_str).parts[-1]
    measurement_path = (
        Path(folder_path)
        .joinpath(f'Matched Measurements/{comparison}/{field}.db')
    )
    con = sql.connect(measurement_path)
    x_df = pd.read_sql(
        "SELECT * FROM x",
        con=con,
        index_col="_time"
    )
    y_df = pd.read_sql(
        "SELECT * FROM y",
        con=con,
        index_col="_time"
    )
    x_df['Fold'] = y_df['Fold']
    for col in x_df.columns:
        if col != "Fold":
            x_df[col] = x_df[col].astype(float)
    for col in y_df.columns:
        if col != "Fold":
            y_df[col] = y_df[col].astype(float)
    x = x_df.to_json(orient='split')
    y = y_df.to_json(orient='split')
    con.close()
    if x is None or y is None:
        return "", ""
    return x, y


@callback(
    Output('calibrated-measurements', 'data'),
    Input('x-measurements', 'data'),
    State('variable-options', 'value'),
    prevent_initial_call=True
)
def calibrate_measurements(x: str, var_path: str) -> str:
    """Calibrate the measurements using provided pipeline."""
    uncal_data = pd.read_json(StringIO(x), orient='split')
    cal_data = pd.DataFrame(index=uncal_data.index)
    pipelines = Path(var_path).glob("*.pkl")
    for pipe_path in pipelines:
        with pipe_path.open('rb') as pipe_bytes:
            pipeline = pickle.load(pipe_bytes)
            for step in pipeline:
                logging.error(step)
            cal_data[
                pipe_path.stem
            ] = pipeline.predict(
                uncal_data.drop(columns=['Fold'], errors='ignore')
            )
    cal_data['Mean'] = cal_data.mean(axis=1)
    cal_data['Fold'] = uncal_data['Fold']
    cal_json = cal_data.to_json(orient='split')
    if cal_json is None:
        return ""
    return cal_json


@callback(
    Output('scatter-graphs', 'figure'),
    Input('calibrated-measurements', 'data'),
    State('y-measurements', 'data'),
    State('field-options', 'value'),
    prevent_initial_call=True
)
def scatter_graphs(
        cal_measurements: str,
        ref_measurements: str,
        var_path: str
    ) -> go.Figure:
    """Generate scatter graphs for validation set."""
    ref = pd.read_json(StringIO(ref_measurements), orient='split')
    cal = pd.read_json(StringIO(cal_measurements), orient='split')

    var = Path(var_path).parts[-1]

    rows = int(np.ceil((cal.shape[1] - 1) / 2))
    titles = sorted(cal.drop(columns='Fold').columns)

    fig = make_subplots(
        rows=rows,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=titles
    )

    for index, fold in enumerate(titles):
        row = (index // 2) + 1
        col = (index % 2 != 0) + 1
        cal_fold = cal[cal['Fold'].isin(['Validation', fold])]
        ref_fold = ref[ref['Fold'].isin(['Validation', fold])]
        fig.add_trace(
            go.Scatter(
                x=cal_fold[str(fold)],
                y=ref_fold[var],
                mode='markers'
            ),
            row=row,
            col=col
        )
        if row == rows:
            fig.update_xaxes(title_text=f"{var} Predicted", row=row, col=col)
        if col == 2:
            fig.update_yaxes(title_text=f"{var} Reference", row=row, col=col)

    fig.update_layout(
        **{
            'width': 750,
            'height': 1125,
            **FIGURE_LAYOUT
        }
    )
    return fig

item_stores = [
    dcc.Store(id="folder-path"),
    dcc.Store(id="x-measurements"),
    dcc.Store(id="y-measurements"),
    dcc.Store(id="calibrated-measurements"),
]

top_row = [
    dbc.Row([html.Div("Graddnodi", className="h1", style={"text-align": "center"})]),
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
                    html.H4("Comparison"),
                    dcc.RadioItems(id="comparison-options", style=checklist_options),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Field"),
                    dcc.RadioItems(id="field-options", style=checklist_options),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Technique"),
                    dcc.RadioItems(
                        id="technique-options", style=checklist_options
                    ),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Scaling Technique"),
                    dcc.RadioItems(id="scaling-options", style=checklist_options),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Variables Used"),
                    dcc.RadioItems(id="variable-options", style=checklist_options),
                ]
            ),
            dbc.Col(
                [
                    html.Button("Run", id='run-button')
                ]
            )
        ]
    )
]

app.layout = dbc.Container(
    [
        *item_stores,
        *top_row,
        html.Hr(),
        *selections,
        dcc.Graph(id='scatter-graphs')
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
