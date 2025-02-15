from io import StringIO
import logging
import os
from pathlib import Path
import pickle
import re
import sqlite3 as sql
from typing import TYPE_CHECKING

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from flask import Flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.pipeline import Pipeline


FULL_TABLE_LAYOUT = {"margin": {"pad": 0, "b": 0, "t": 1, "l": 0, "r": 0}}

FIGURE_LAYOUT = {
    "margin": {"pad": 0, "b": 40, "t": 50, "l": 40, "r": 0},
    "showlegend": True,
    "font": {"size": 10},
    "paper_bgcolor": "rgba(0,0,0,0)",
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
        "y": 1.10
    }
}

RENAME_OPTIONS = {
    "PM2.5": r"\\text{PM}_{2.5}",
    "PM10": r"\\text{PM}_{10}",
    "NO2": r"\\text{NO}_{2}",
    "O3": r"\\text{O}_{3}",
    r"NO[^\}]": r"\\text{NO} ",
    "^2": r"^{2}",
    "RH T": "RH*T",
    "RH": r"\\text{RH}",
    "T ": r"\\text{T} ",
    "Time Since Origin": r"\\text{Time Since Origin}",
    ", (.*)": r"\\text{, \g<1>}",
}

MARKERS = [
    "x-thin-open",
    "cross-thin-open",
    "asterisk-open",
    "circle-open-dot",
    "square-open-dot",
    "star-triangle-up-open-dot"
]

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
        for i in sorted(path.glob("*"))
        if i.is_dir()
    ]


@callback(
    Output("comparison-options", "options"),
    Input("folder-path", "data"),
    prevent_initial_call=True,
)
def comparison_opts(path_str: str) -> list[dict[str, str]]:
    """Get all comparison directories."""
    path = Path(path_str).joinpath("Pipelines")
    return get_glob(path)


@callback(
    Output("field-options", "options"),
    Input("comparison-options", "value"),
    prevent_initial_call=True,
)
def field_opts(path_str: str) -> list[dict[str, str]]:
    """Get all field directories."""
    path = Path(path_str)
    return get_glob(path)


@callback(
    Output("technique-options", "options"),
    Input("field-options", "value"),
    prevent_initial_call=True,
)
def tech_opts(path_str: str) -> list[dict[str, str]]:
    """Get all technique directories."""
    path = Path(path_str)
    return get_glob(path)


@callback(
    Output("scaling-options", "options"),
    Input("technique-options", "value"),
    prevent_initial_call=True,
)
def scaling_opts(path_str: str) -> list[dict[str, str]]:
    """Get all scaling directories."""
    path = Path(path_str)
    return get_glob(path)


@callback(
    Output("variable-options", "options"),
    Input("scaling-options", "value"),
    prevent_initial_call=True,
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
    prevent_initial_call=True,
)
def get_data(
    _: int, folder_path: str, comparison_str: str, field_str: str
) -> tuple[str, str]:
    """Get comparison measurements."""
    comparison = Path(comparison_str).parts[-1]
    field = Path(field_str).parts[-1]
    measurement_path = Path(folder_path).joinpath(
        f"Matched Measurements/{comparison}/{field}.db"
    )
    con = sql.connect(measurement_path)
    x_df = pd.read_sql("SELECT * FROM x", con=con, index_col="_time")
    y_df = pd.read_sql("SELECT * FROM y", con=con, index_col="_time")
    x_df["Fold"] = y_df["Fold"]
    for col in x_df.columns:
        if col != "Fold":
            x_df[col] = x_df[col].astype(float)
    for col in y_df.columns:
        if col != "Fold":
            y_df[col] = y_df[col].astype(float)
    x = x_df.to_json(orient="split")
    y = y_df.to_json(orient="split")
    con.close()
    if x is None or y is None:
        return "", ""
    return x, y


@callback(
    Output("calibrated-measurements", "data"),
    Input("x-measurements", "data"),
    State("variable-options", "value"),
    prevent_initial_call=True,
)
def calibrate_measurements(x: str, var_path: str) -> str:
    """Calibrate the measurements using provided pipeline."""
    uncal_data = pd.read_json(StringIO(x), orient="split")
    cal_data = pd.DataFrame(index=uncal_data.index)
    pipelines = Path(var_path).glob("*.pkl")
    for pipe_path in pipelines:
        with pipe_path.open("rb") as pipe_bytes:
            pipeline = pickle.load(pipe_bytes)  # noqa: S301
            cal_data[pipe_path.stem] = pipeline.predict(
                uncal_data.drop(columns=["Fold"], errors="ignore")
            )
    cal_data["Mean"] = cal_data.mean(axis=1)
    cal_data["Fold"] = uncal_data["Fold"]
    cal_json = cal_data.to_json(orient="split")
    if cal_json is None:
        return ""
    return cal_json


def get_shap(
        x: pd.DataFrame,
        y: pd.DataFrame,
        pipelines: dict[int, Pipeline]
    ) -> pd.DataFrame:
    """Get shap values from model."""
    shaps_list = []
    for fold, pipeline in pipelines.items():
        if len(pipelines.keys()) > 1:
            fold_index = y[y.loc[:, "Fold"].isin(["Validation"])].index
            x_data = x.loc[fold_index, :]
        else:
            x_data = x
        predicted = pd.Series(
            pipeline.predict(x_data).flatten(),
            index=x_data.index
        )
        x_data = x_data[predicted.notna()]
        logging.critical(x_data)

        transformed_data = pipeline[:-1].transform(x_data)
        if transformed_data.shape[0] > 100:
            transformed_data = transformed_data.sample(n=100)

        logging.critical(transformed_data)
        logging.critical([(fold, col.replace("selector__", "")) for col in pipeline[:-1].get_feature_names_out()])

        explainer = shap.KernelExplainer(
            model=pipeline[-1].predict,
            data=transformed_data,
            link="identity"
        )
        shaps = explainer.shap_values(transformed_data)
        shaps_fold = pd.DataFrame(
            shaps,
            index=transformed_data.index,
            columns=pd.MultiIndex.from_tuples(
                [(fold, col.replace("selector__", "")) for col in pipeline[:-1].get_feature_names_out()]
            )
        )
        shaps_list.append(shaps_fold)
    return pd.concat(shaps_list, axis=1).sort_index()


def shap_graph(
    x: pd.DataFrame,
    y: pd.DataFrame,
    path: Path
) -> go.Figure:
    """Plot shap values."""
    shap_path = path.joinpath('shaps.tar.gz')
    shap_combined = {}
    pipelines = {}
    pipelines_paths = path.glob("*.pkl")
    for pipe in pipelines_paths:
        with pipe.open('rb') as pickle_file:
            pipelines[pipe.stem] = pickle.load(pickle_file)
    try:
        shaps = pd.read_pickle(shap_path)
    except FileNotFoundError:
        shaps = get_shap(x, y, pipelines)
        shaps.to_pickle(shap_path)
    for fold in shaps.columns.get_level_values(0):
        for feature, col in shaps.xs(fold, axis=1, level=0).items():
            if feature not in shap_combined:
                shap_combined[feature] = {}
            scatter_df = pd.DataFrame()
            scatter_df["Shap"] = col
            transformed_data = pipelines[fold][:-1].transform(x)
            scatter_df["Measurement"] = transformed_data.loc[
                scatter_df.index, feature
            ]
            scatter_df = scatter_df.dropna()
            shap_combined[feature][fold] = scatter_df
    folds_plotted = set()

    rows = int(np.ceil(len(shap_combined.keys()) / 2))
    shap_titles = []
    for tit in shap_combined.keys():
        math_mode = False
        for pat, repl in RENAME_OPTIONS.items():
            tit = re.sub(pat, repl, tit)
            math_mode = True
        shap_titles.append(f"{'$' if math_mode else ''}{tit}{'$' if math_mode else ''}")
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=list(shap_titles),
        vertical_spacing=0.05,
        # shared_xaxes=True,
        # shared_yaxes=True
    )
    for title_index, (title, graph_data) in enumerate(shap_combined.items()):
        max_measurement_vals = []
        math_mode = False
        for pat, repl in RENAME_OPTIONS.items():
            title = re.sub(pat, repl, title)
            math_mode = True
        for _, graph_df in sorted(graph_data.items()):
            max_measurement_vals.append(graph_df["Measurement"].max()) 

        for index, (fold, graph_df) in enumerate(sorted(graph_data.items())):
            symbol_index = index % len(MARKERS)
            row = (title_index // 2) + 1
            col = (title_index % 2 != 0) + 1
            fig.add_trace(
                go.Scatter(
                    x=graph_df['Measurement'].div(max(max_measurement_vals)),
                    y=graph_df['Shap'],
                    mode="markers",
                    marker_symbol=MARKERS[symbol_index],
                    name=fold,
                    showlegend=fold not in folds_plotted
                ),
                row=row,
                col=col
            )
            fig.layout[f"xaxis{title_index+1}"]["title"] = (
                f"{'$' if math_mode else ''}"
                f"{title} / max({title})"
                f"{'$' if math_mode else ''}"
            )
            fig.layout[f"yaxis{title_index+1}"]["title"] = "Influence on prediction"
            if len(graph_data) > 1:
                fig.add_annotation(
                    xref='x domain',
                    yref='y domain',
                    x=0.01,
                    y=0.97,
                    text=f'({chr(97+title_index)})',
                    font={"size": 24},
                    showarrow=False,
                    row=row,
                    col=col
                )
            folds_plotted.update(fold)


    fig.update_layout(**{"width": 750, "height": 375 * rows, **FIGURE_LAYOUT})


    return fig


def scatter_graph(
    x: pd.DataFrame, y: pd.DataFrame, var: str, x_type: str, y_type: str
) -> go.Figure:
    """Scatter graphs."""
    rows = int(np.ceil((x.shape[1] - 1) / 2))
    titles = sorted(
        [*[c for c in x.loc[:, "Fold"].unique() if c != "Validation"], "Mean"]
    )

    fig = make_subplots(
        rows=rows,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=titles,
    )

    for index, fold in enumerate(titles):
        row = (index // 2) + 1
        col = (index % 2 != 0) + 1
        x_fold = x[x["Fold"].isin(["Validation", fold])].drop(columns='Fold')
        y_fold = y[y["Fold"].isin(["Validation", fold])].drop(columns='Fold')
        fig.add_trace(
            go.Scatter(x=x_fold[str(fold)], y=y_fold[var], mode="markers"),
            row=row,
            col=col,
        )
        if row == rows:
            fig.update_xaxes(title_text=f"{var} {x_type}", row=row, col=col)
        if col == 2:
            fig.update_yaxes(title_text=f"{var} {y_type}", row=row, col=col)

    fig.update_layout(**{"width": 750, "height": 375 * rows, **FIGURE_LAYOUT})
    return fig


@callback(
    Output("scatter-graphs", "figure"),
    Input("graph-type", "value"),
    Input("calibrated-measurements", "data"),
    State("x-measurements", "data"),
    State("y-measurements", "data"),
    State("field-options", "value"),
    State("variable-options", "value"),
    prevent_initial_call=True,
)
def graphs(
    graph_type: str,
    cal_measurements: str,
    uncal_measurements: str,
    ref_measurements: str,
    var_path: str,
    pipe_path: str
) -> go.Figure:
    """Generate scatter graphs for validation set."""
    ref = pd.read_json(StringIO(ref_measurements), orient="split")
    cal = pd.read_json(StringIO(cal_measurements), orient="split")
    uncal = pd.read_json(StringIO(uncal_measurements), orient="split")

    var = Path(var_path).parts[-1]

    if graph_type == "cal-v-ref":
        return scatter_graph(cal, ref, var, "Predicted", "Reference")
    if graph_type == "cal-v-uncal":
        return scatter_graph(cal, uncal, var, "Predicted", "Uncalibrated")
    if graph_type == "shap-graphs":
        return shap_graph(uncal, ref, Path(pipe_path))
    return go.Figure()


item_stores = [
    dcc.Store(id="folder-path"),
    dcc.Store(id="x-measurements"),
    dcc.Store(id="y-measurements"),
    dcc.Store(id="calibrated-measurements"),
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
                    html.H4("Comparison"),
                    dcc.RadioItems(
                        id="comparison-options", style=checklist_options
                    ),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Field"),
                    dcc.RadioItems(
                        id="field-options", style=checklist_options
                    ),
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
                    dcc.RadioItems(
                        id="scaling-options", style=checklist_options
                    ),
                ]
            ),
            dbc.Col(
                [
                    html.H4("Variables Used"),
                    dcc.RadioItems(
                        id="variable-options", style=checklist_options
                    ),
                ]
            ),
            dbc.Col([html.Button("Run", id="run-button")]),
        ]
    )
]

graph_tab = [
    dcc.Tabs(
        id="graph-type",
        value="cal-v-ref",
        children=[
            dcc.Tab(label="Calibrated vs Reference", value="cal-v-ref"),
            dcc.Tab(label="Calibrated vs Uncalibrated", value="cal-v-uncal"),
            dcc.Tab(label="Shaps", value="shap-graphs"),
        ],
    )
]

app.layout = dbc.Container(
    [
        *item_stores,
        *top_row,
        html.Hr(),
        *selections,
        *graph_tab,
        dcc.Graph(mathjax=True, id="scatter-graphs"),
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
