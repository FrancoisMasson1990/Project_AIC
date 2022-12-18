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

import bz2
import pickle
import warnings

import dash
import dash_bootstrap_components as dbc
from dash import Dash
from dash import ctx
from dash import dcc
from dash import html
from dash import no_update
from flask import Flask

import aic.viewer.files as fs
import aic.viewer.web_visual as vs
from aic.dashboard.template.config_dragmode import config_mode_bar

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
                            dcc.Store(id="annotation_data", data={}),
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
                                    html.Button(
                                        "Home",
                                        id="home_btn",
                                        n_clicks=0,
                                        style={
                                            "marginLeft": "0px",
                                            "width": "130px",
                                        },
                                    ),
                                ]
                            ),
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                dbc.Col(
                                                    [
                                                        html.P("Annotations"),
                                                        dcc.Download(
                                                            id="download_pred"
                                                        ),
                                                        html.Button(
                                                            "Download",
                                                            id="download_btn",
                                                            n_clicks=0,
                                                            style={
                                                                "marginLeft": "0px",
                                                                "marginBottom": "10px",
                                                                "width": "120px",
                                                            },
                                                        ),
                                                        html.Pre(
                                                            id="download_data"
                                                        ),
                                                        dbc.Tooltip(
                                                            "Download all the annonations present in the images.",
                                                            target="download_btn",
                                                        ),
                                                        html.P(
                                                            "Current frame",
                                                        ),
                                                        html.Button(
                                                            "Save",
                                                            id="frame_btn",
                                                            n_clicks=0,
                                                            style={
                                                                "marginLeft": "0px",
                                                                "marginBottom": "10px",
                                                                "width": "120px",
                                                            },
                                                        ),
                                                        html.Pre(
                                                            id="frame_data"
                                                        ),
                                                        dbc.Tooltip(
                                                            "Save all the drawings in the current image.",
                                                            target="frame_btn",
                                                        ),
                                                        html.P("Shapes"),
                                                        html.Button(
                                                            "Clear",
                                                            id="clear_btn",
                                                            n_clicks=0,
                                                            style={
                                                                "marginLeft": "0px",
                                                                "marginBottom": "10px",
                                                                "width": "120px",
                                                            },
                                                        ),
                                                        dbc.Tooltip(
                                                            "Clear all the drawings in the images.",
                                                            target="clear_btn",
                                                        ),
                                                    ],
                                                    align="center",
                                                )
                                            ),
                                        ]
                                    ),
                                ],
                            ),
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
                                                        config=config_mode_bar,
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
        dash.Input(
            component_id="agatston-graph-2d", component_property="figure"
        ),
        dash.Input(component_id="upload-data", component_property="contents"),
        dash.State(component_id="upload-data", component_property="filename"),
        dash.State(
            component_id="upload-data", component_property="last_modified"
        ),
        dash.Input(component_id="clear_btn", component_property="n_clicks"),
    )
    def update_output_2d(fig, contents, names, dates, n_click):
        """Update 2d graphes."""
        fs.rm_tmp_folders()
        fs.rm_tmp_files()
        triggered_id = ctx.triggered_id
        if not fig:
            return vs.init_graph_2d(fig)
        if "data" in fig.keys():
            # Case of init graph
            if not fig["data"]:
                output = vs.parse_dcm(contents, names, dates)
                fig = vs.labeling_graph_2d(fig, output)
                fig = vs.set_layout_mode(fig, dragmode="drawopenpath")
        if triggered_id == "clear_btn":
            # Case of clear all annotations
            if isinstance(fig, dict) and "layout" in fig.keys():
                layout = fig["layout"]
                if "shapes" in layout.keys():
                    fig["layout"]["shapes"] = []
        return fig

    @dash_app.callback(
        dash.Output("hidden_redirect_callback", "children"),
        dash.Input("home_btn", "n_clicks"),
    )
    def redirect_home(n_clicks):
        if n_clicks > 0:
            return dcc.Location(pathname="/", id="redirect_home")

    @dash_app.callback(
        dash.Output("download_data", "children"),
        dash.Input("annotation_data", component_property="data"),
        dash.Input(component_id="download_btn", component_property="n_clicks"),
        prevent_initial_call=True,
    )
    def download_annotation(data, n_clicks):
        fs.rm_tmp_folders()
        fs.rm_tmp_files()
        triggered_id = ctx.triggered_id
        file_name = "./cache/annotation.pbz2"
        if triggered_id == "download_btn":
            if data:
                breakpoint()
                with bz2.BZ2File(file_name, "wb") as f:
                    pickle.dump(data, f)
                return dcc.send_file(file_name)
        return no_update

    @dash_app.callback(
        dash.Output("annotation_data", component_property="data"),
        dash.Input("annotation_data", component_property="data"),
        dash.Input(
            component_id="agatston-graph-2d", component_property="relayoutData"
        ),
        dash.Input(component_id="frame_btn", component_property="n_clicks"),
        dash.State(
            component_id="agatston-graph-2d", component_property="figure"
        ),
        prevent_initial_call=True,
    )
    def save_annotation(data, relayoutData, n_clicks, fig):
        triggered_id = ctx.triggered_id
        if triggered_id == "frame_btn":
            # Case of clear all annotations
            if isinstance(fig, dict) and "layout" in fig.keys():
                if "sliders" in fig["layout"].keys():
                    slide = fig["layout"]["sliders"][0]["active"]
                    # Need to extract point inside
                    data[slide] = relayoutData
                    return data
        return no_update

    return dash_app
