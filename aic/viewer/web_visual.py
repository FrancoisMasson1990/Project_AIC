#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for the UX Viewer
"""

import base64
import bz2
import io
import pickle

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

import aic.misc.files as fs
import aic.model.inference as infer
import aic.processing.scoring as sc


def generate_imgs(data, index, fig):
    """Generate images."""
    imgs = data["image"][index]
    # Update Image
    for i in tqdm(range(imgs.shape[0])):
        img = imgs[i]
        fig.add_trace(
            go.Heatmap(
                z=img,
                hoverongaps=False,
                hoverinfo="skip",
                colorscale="gray",
                showscale=False,
                zauto=True,
            )
        )
        fig.data[i].visible = False

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(px.imshow(imgs[0]).layout)
    fig.update_traces(
        hovertemplate="x: %{x} <br> y: %{y} <br> Hounsfield unit: %{z}"
    )
    fig.update_coloraxes(showscale=False)

    return fig


def generate_mask(data, index, fig):
    """Generate masks."""
    imgs = data["image"][index]
    masks = data["mask_agatston"][index]
    area = data["area"]
    threshold_min = data["threshold_min"]
    threshold_max = data["threshold_max"]
    # Update prediction
    for i in tqdm(range(masks.shape[0])):
        prediction = imgs[i].copy()
        mask = masks[i]
        prediction[mask == 0] = 0
        if threshold_min is not None:
            prediction[prediction < threshold_min] = 0
        if threshold_max is not None:
            prediction[prediction > threshold_max] = 0
        prediction[prediction > 0] = 1
        area_, lw = sc.area_measurements(prediction)
        for j, number_of_pix in enumerate(area_):
            if j != 0:
                # (density higher than 1mm2)
                if number_of_pix * area <= 1:
                    prediction[lw == j] = 0
        prediction = prediction.astype("float")
        prediction[prediction < 1] = np.nan
        prediction[prediction >= 1] = 1
        fig.add_trace(
            go.Heatmap(
                z=prediction,
                coloraxis="coloraxis1",
                hoverongaps=False,
                hoverinfo="skip",
            )
        )
        fig.data[i + masks.shape[0]].visible = False
    return fig


def update_graph_2d(data, zmin=130, zmax=600):
    """Draw prediction graphes."""
    fig = go.Figure()
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if data:
        index = []
        for z in range(data["mask_agatston"].shape[0]):
            # Check if all 2D numpy array contains only 0
            result = np.all((data["mask_agatston"][z] == 0))
            if not result:
                index.append(z)
        index = np.array(index)
        extra_index = 5
        index = np.arange(index[0] - extra_index, index[-1] + extra_index)

        fig = generate_imgs(data, index, fig)
        fig = generate_mask(data, index, fig)

        # Visibility
        fig.data[0].visible = True
        fig.data[len(fig.data) // 2].visible = True

        steps = []
        for i in range(len(fig.data) // 2):
            step = dict(
                method="restyle",
                args=[{"visible": [False] * len(fig.data)}],
            )
            # Toggle i'th trace to "visible"
            step["args"][0]["visible"][i] = True
            # Toggle i'th trace to "visible"
            step["args"][0]["visible"][i + len(fig.data) // 2] = True
            steps.append(step)

        sliders = [
            dict(
                active=0, currentvalue={"prefix": "Dicom file: "}, steps=steps
            )
        ]

        fig.update_layout(
            sliders=sliders,
            coloraxis1=dict(colorscale="Reds", showscale=False),
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {
                                        "colorscale": "Greys",
                                        "showscale": False,
                                        "zauto": True,
                                    }
                                ],
                                label="Original",
                                method="restyle",
                            ),
                            dict(
                                args=[
                                    {
                                        "colorscale": "Jet",
                                        "showscale": True,
                                        "zmax": zmax,
                                        "zmin": zmin,
                                    }
                                ],
                                label="Color Scale",
                                method="restyle",
                            ),
                        ]
                    ),
                    type="buttons",
                    direction="right",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="right",
                    yanchor="top",
                    font=dict(color="#000000"),
                )
            ],
        )
    fig.update_layout(template="plotly_dark")
    fig.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")
    return fig


def update_graph_3d(data, cmin=130, cmax=600):
    """Draw prediction graphes."""
    fig = go.Figure()
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if data:
        if "valve" in data.keys():
            x = data["valve"][:, 0]
            y = data["valve"][:, 1]
            z = data["valve"][:, 2]
            c_valve = data["valve"][:, 3]

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(size=3, color="white", opacity=0.6),
                )
            )
        if "candidate" in data.keys():
            x = data["candidate"][:, 0]
            y = data["candidate"][:, 1]
            z = data["candidate"][:, 2]

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(color="red", size=3, opacity=0.6),
                )
            )

        fig.update_scenes(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
        )
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {
                                        "marker": {
                                            "color": "white",
                                            "size": 3,
                                            "opacity": 0.6,
                                            "showscale": False,
                                        },
                                    },
                                    [0],
                                ],
                                label="Original",
                                method="restyle",
                            ),
                            dict(
                                args=[
                                    {
                                        "marker": {
                                            "color": c_valve,
                                            "colorscale": "Jet",
                                            "size": 3,
                                            "opacity": 0.6,
                                            "showscale": True,
                                            "cmin": cmin,
                                            "cmax": cmax,
                                        },
                                    },
                                    [0],
                                ],
                                label="Color Scale",
                                method="restyle",
                            ),
                        ]
                    ),
                    type="buttons",
                    direction="right",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="right",
                    yanchor="top",
                    font=dict(color="#000000"),
                )
            ],
        )
    fig.update_layout(template="plotly_dark")
    fig.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")
    return fig


def parse_contents(model, online, contents, filenames, dates):
    """Parse content."""
    response = None
    if not filenames:
        return response

    if len(filenames) > 1:
        data = []
        files_types = []
        for f, c in zip(filenames, contents):
            if f.endswith(".dcm") or f.endswith(".txt"):
                data.append(c)
                files_types.append(f)
        if data:
            # Should send back dict with same element has the
            # predictions.pbz2
            config = fs.get_configs_root() / "web_config.yml"
            response = infer.get_inference(
                data, files_types, str(config), model, online
            )
            if not response:
                return None
    else:
        contents = contents[0]
        filenames = filenames[0]
        content_type, content_string = contents.split(",")
        if filenames.endswith(".pbz2"):
            decoded = base64.b64decode(content_string)
            decoded = io.BytesIO(decoded)
            with bz2.BZ2File(decoded, "rb") as f:
                response = pickle.load(f)
    return response


def two_years_model(coefficient):
    """Get 2 years model."""
    out = 0
    values = list(coefficient.values())
    if any(elem is None for elem in values):
        return ""
    for key, val in coefficient.items():
        if val == "Yes":
            val = 1
        elif val == "No":
            val = 0
        elif val == "Man":
            val = 1
        elif val == "Woman":
            val = 0
        if key == "age":
            out += float(val) * 0
        elif key == "sex":
            out += -float(val) * (0.1)
        elif key == "hypertension":
            out += float(val) * (0)
        elif key == "renal":
            out += float(val) * (0.5)
        elif key == "implantation":
            out += float(val) * (0.25)
        elif key == "cholesterol":
            out += float(val) * (0.04)
        elif key == "ldl":
            out += float(val) * (0.07)
        elif key == "gradient":
            out += float(val) * (0.002)
        elif key == "size":
            out += -float(val) * (0.01)
        elif key == "score":
            out += float(val) * (0.08)

    return int(np.round(out, 0))


def five_years_model(coefficient):
    """Get 5 years model."""
    out = 0
    values = list(coefficient.values())
    if any(elem is None for elem in values):
        return ""
    for key, val in coefficient.items():
        if val == "Yes":
            val = 1
        elif val == "No":
            val = 0
        elif val == "Man":
            val = 1
        elif val == "Woman":
            val = 0
        if key == "age":
            out += float(val) * 0
        elif key == "sex":
            out += -float(val) * (0.2)
        elif key == "hypertension":
            out += float(val) * (0)
        elif key == "renal":
            out += float(val) * (0.5)
        elif key == "implantation":
            out += float(val) * (0.28)
        elif key == "cholesterol":
            out += float(val) * (0.03)
        elif key == "ldl":
            out += float(val) * (0.02)
        elif key == "gradient":
            out += float(val) * (0.5)
        elif key == "size":
            out += -float(val) * (0.01)
        elif key == "score":
            out += float(val) * (0.15)

    return int(np.round(out, 0))


def operations(coeff_dict):
    """Estimate operation score."""
    output_1 = two_years_model(coeff_dict)
    output_2 = five_years_model(coeff_dict)
    if isinstance(output_1, int) and isinstance(output_2, int):
        return [output_1, output_2]
    else:
        message = "Missing informations for Clinical Outcomes"
        return message
