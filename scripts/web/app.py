#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Code for web deployment
"""

import aic.viewer.web_visual as vs
import aic.viewer.files as fs
import bz2
import pickle
import warnings
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash

warnings.filterwarnings("ignore")


fs.mk_tmp_folder()
fs.rm_tmp_folders()
fs.rm_tmp_files()
tab_style = {"backgroundColor": "#1e1e1e", "color": "white"}
# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Agatston Score Dashboard"
app._favicon = "heart.ico"
app.config.suppress_callback_exceptions = True
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        # Patient Name
                        html.P(
                            [
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select Files")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                    },
                                    multiple=True,
                                ),
                            ]
                        ),
                        html.P(
                            [
                                dcc.Download(id="save_pred"),
                                html.Button("Save results", id="download"),
                            ]
                        ),
                        html.H2(id="patient_name"),
                        # Outcomes
                        html.P(id="outcomes"),
                        # Hyperparameters
                        html.P("Agatston Score"),
                        # Agatston score
                        html.P(
                            [
                                dcc.Input(
                                    id="score", value="", type="text", disabled=True
                                )
                            ]
                        ),
                        # Age
                        html.P("Age"),
                        html.P(
                            [
                                dcc.Input(
                                    id="age", value=None, type="text", disabled=False
                                )
                            ]
                        ),
                        # Sex
                        html.P("Sex"),
                        html.P(
                            [
                                dcc.Dropdown(
                                    options=["Man", "Woman"], value=None, id="sex"
                                )
                            ]
                        ),
                        # Hypertension
                        html.P("Hypertension"),
                        html.P(
                            [
                                dcc.Dropdown(
                                    options=["Yes", "No"], value=None, id="hypertension"
                                )
                            ]
                        ),
                        # Renal Disease
                        html.P("Renal Disease"),
                        html.P(
                            [
                                dcc.Dropdown(
                                    options=["Yes", "No"], value=None, id="renal"
                                )
                            ]
                        ),
                        # Years since Implantation
                        html.P("Years Since Implantation"),
                        html.P(
                            [
                                dcc.Input(
                                    id="implantation",
                                    value=None,
                                    type="text",
                                    disabled=False,
                                )
                            ]
                        ),
                        # Total cholesterol
                        html.P("Total Cholesterol"),
                        html.P(
                            [
                                dcc.Input(
                                    id="cholesterol",
                                    value=None,
                                    type="text",
                                    disabled=False,
                                )
                            ]
                        ),
                        # LDL Cholesterol
                        html.P("LDL Cholesterol"),
                        html.P(
                            [
                                dcc.Input(
                                    id="ldl", value=None, type="text", disabled=False
                                )
                            ]
                        ),
                        # Mean Gradient
                        html.P("Mean Gradient"),
                        html.P(
                            [
                                dcc.Input(
                                    id="gradient",
                                    value=None,
                                    type="text",
                                    disabled=False,
                                )
                            ]
                        ),
                        # Prosthesis Size
                        html.P("Prosthesis Size"),
                        html.P(
                            [
                                dcc.Dropdown(
                                    options=["19", "21", "23", "25", "29"],
                                    value=None,
                                    id="size",
                                )
                            ]
                        ),
                    ],
                ),
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    label="2D View",
                                    children=[
                                        dcc.Graph(
                                            id="agatston-graph-2d",
                                            style={"height": "100vh"},
                                        ),
                                        dbc.Progress(
                                            id="pbar", striped=True, animated=True
                                        ),
                                        dcc.Interval(
                                            id="timer_progress", interval=1000
                                        ),
                                    ],
                                    style=tab_style,
                                    selected_style=tab_style,
                                ),
                                dcc.Tab(
                                    label="3D View",
                                    children=[
                                        dcc.Graph(
                                            id="agatston-graph-3d",
                                            style={"height": "100vh"},
                                        )
                                    ],
                                    style=tab_style,
                                    selected_style=tab_style,
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ]
)


@app.callback(
    dash.Output(component_id="agatston-graph-2d", component_property="figure"),
    dash.Output(component_id="agatston-graph-3d", component_property="figure"),
    dash.Output(component_id="score", component_property="value"),
    dash.Output(component_id="patient_name", component_property="children"),
    dash.Input(component_id="upload-data", component_property="contents"),
    dash.State(component_id="upload-data", component_property="filename"),
    dash.State(component_id="upload-data", component_property="last_modified"),
)
def update_output_2d(contents, names, dates):
    """Update 2d graphes."""
    fs.rm_tmp_folders()
    fs.rm_tmp_files()
    output = vs.parse_contents(contents, names, dates)
    fig_2d = vs.update_graph_2d(output)
    fig_3d = vs.update_graph_3d(output)
    score = None
    string = "Project Valve AIC For Patient : "
    if output and "score" in output.keys():
        score = "{0:.4f}".format(output["score"])
    if output and "data_path" in output.keys():
        name = output["data_path"].split("/")[0]
        string += name
    if output:
        with bz2.BZ2File("./cache/prediction.pbz2", "wb") as f:
            pickle.dump(output, f)
    return fig_2d, fig_3d, score, string


@app.callback(
    dash.Output(component_id="outcomes", component_property="children"),
    dash.Input(component_id="age", component_property="value"),
    dash.Input(component_id="score", component_property="value"),
    dash.Input(component_id="sex", component_property="value"),
    dash.Input(component_id="renal", component_property="value"),
    dash.Input(component_id="gradient", component_property="value"),
    dash.Input(component_id="size", component_property="value"),
    dash.Input(component_id="ldl", component_property="value"),
    dash.Input(component_id="cholesterol", component_property="value"),
    dash.Input(component_id="implantation", component_property="value"),
)
def update_outcome(
    age,
    score,
    sex,
    renal,
    gradient,
    size,
    ldl,
    cholesterol,
    implantation,
):
    """Get outputs."""
    values = [age, score, sex, renal, gradient, size, ldl, cholesterol, implantation]
    if any(elem is None for elem in values):
        return "Missing informations for Clinical Outcomes"
    elif any(not elem for elem in values):
        return "Missing informations for Clinical Outcomes"
    else:
        coeff_dict = {}
        coeff_dict["age"] = age
        coeff_dict["score"] = score
        coeff_dict["sex"] = sex
        coeff_dict["renal"] = renal
        coeff_dict["gradient"] = gradient
        coeff_dict["size"] = size
        coeff_dict["ldl"] = ldl
        coeff_dict["cholesterol"] = cholesterol
        coeff_dict["implantation"] = implantation
        response = vs.operations(coeff_dict)
        if isinstance(response, list):
            return (
                f"Risque of reoperation after 2 years"
                + f" is {response[0]}"
                + f" and after 5 years is {response[1]}"
            )
        elif isinstance(response, str):
            return response
        else:
            return "Error"


@app.callback(
    dash.Output("save_pred", "data"),
    dash.Input("download", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    """Download content."""
    if isinstance(n_clicks, int):
        return dcc.send_file("./cache/prediction.pbz2")


@app.callback(
    dash.Output("pbar", "value"),
    dash.Output("pbar", "label"),
    dash.Input("timer_progress", "n_intervals"),
    prevent_initial_call=True,
)
def callback_progress(n_intervals):
    """Update progress bar."""
    try:
        with open("./cache/progress.txt", "r") as file:
            str_raw = file.read()
        last_line = list(filter(None, str_raw.split("\n")))[-1]
        percent = float(last_line.split("%")[0])
    except Exception as e:
        percent = 0
    finally:
        text = f"{percent:.0f}%"
        return percent, text


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=True)
