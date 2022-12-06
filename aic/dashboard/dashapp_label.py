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

import warnings

import dash
import dash_bootstrap_components as dbc
from dash import Dash
from dash import dcc
from dash import html
from flask import Flask

import aic.dashboard.template.annotated_card as a_card
import aic.viewer.files as fs
import aic.viewer.web_visual as vs

warnings.filterwarnings("ignore")


fs.mk_tmp_folder()
fs.rm_tmp_folders()
fs.rm_tmp_files()


# Initialize the app
def create_dash_label(
    server: Flask,
    url_base_pathname: str = None,
    model=None,
    online: bool = True,
) -> Dash:
    tab_style = {"backgroundColor": "#1e1e1e", "color": "white"}
    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname=url_base_pathname,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )
    dash_app.title = "Agatston Score Dashboard"
    dash_app._favicon = "heart.ico"
    dash_app.config.suppress_callback_exceptions = True
    dash_app.layout = html.Div(
        children=[
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="four columns div-user-controls",
                        children=[
                            html.Div(id="hidden_redirect_callback"),
                            # Patient Name
                            html.P(
                                [
                                    dcc.Upload(
                                        id="upload-data",
                                        children=html.Div(
                                            [
                                                "Drag and Drop or ",
                                                html.A("Select Files"),
                                            ]
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
                                    html.Button(
                                        "Save results",
                                        id="download_btn",
                                        n_clicks=0,
                                        style={"marginLeft": "0px"},
                                    ),
                                    html.Button(
                                        "Clear results",
                                        id="clear_btn",
                                        n_clicks=0,
                                        style={"marginLeft": "10px"},
                                    ),
                                    html.Button(
                                        "Home",
                                        id="home_btn",
                                        n_clicks=0,
                                        style={"marginLeft": "10px"},
                                    ),
                                ]
                            ),
                            a_card.annotated_data_card(),
                        ],
                    ),
                    html.Div(
                        className="eight columns div-for-charts bg-grey",
                        children=[
                            dcc.Tabs(
                                [
                                    dcc.Tab(
                                        label="Labeling DCM Images",
                                        children=[
                                            dcc.Loading(
                                                id="loading-1",
                                                type="circle",
                                                children=[
                                                    dcc.Graph(
                                                        id="agatston-graph-2d",
                                                        style={
                                                            "height": "100vh"
                                                        },
                                                    )
                                                ],
                                            ),
                                        ],
                                        style=tab_style,
                                        selected_style=tab_style,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )

    @dash_app.callback(
        dash.Output(
            component_id="agatston-graph-2d", component_property="figure"
        ),
        dash.Input(component_id="clear_btn", component_property="n_clicks"),
        dash.Input(component_id="upload-data", component_property="contents"),
        dash.State(component_id="upload-data", component_property="filename"),
        dash.State(
            component_id="upload-data", component_property="last_modified"
        ),
    )
    def update_output_2d(clear_clicks, contents, names, dates):
        """Update 2d graphes."""
        fs.rm_tmp_folders()
        fs.rm_tmp_files()
        output = vs.parse_dcm(contents, names, dates)
        fig_2d = vs.labeling_graph_2d(output)
        return fig_2d

    @dash_app.callback(
        dash.Output("hidden_redirect_callback", "children"),
        dash.Input("home_btn", "n_clicks"),
    )
    def redirect_home(n_clicks):
        if n_clicks > 0:
            return dcc.Location(pathname="/", id="redirect_home")

    return dash_app
