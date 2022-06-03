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

import sys
import aic.model.inference as infer
import pickle
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import aic.processing.scoring as sc
import aic.misc.files as fs
from tqdm import tqdm
import bz2
import base64
import io


def generate_imgs(data, index, fig):
    """Generate images."""
    imgs = data["image"][index]
    # Update Image
    for i in tqdm(range(imgs.shape[0])):
        img = imgs[i]
        fig.add_trace(px.imshow(img,
                                color_continuous_scale='gray').data[0])
        fig.data[i].visible = False

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(px.imshow(imgs[0],
                                color_continuous_scale='gray').layout)
    fig.update_traces(
        hovertemplate="x: %{x} <br> y: %{y} <br> Hounsfield unit: %{z}")
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
                if number_of_pix*area <= 1:
                    prediction[lw == j] = 0
        prediction = prediction.astype('float')
        prediction[prediction < 1] = np.nan
        prediction[prediction >= 1] = 1
        fig.add_trace(go.Heatmap(z=prediction,
                                 hoverongaps=False,
                                 colorscale="Reds",
                                 hoverinfo="skip",
                                 showscale=False))
        fig.data[i+masks.shape[0]].visible = False
    return fig


def update_graph_2d(data):
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
        index = np.arange(index[0]-extra_index,
                          index[-1]+extra_index)

        std_err_backup = sys.stderr
        file_prog = open('./cache/progress.txt', 'w')
        sys.stderr = file_prog
        fig = generate_imgs(data, index, fig)
        fig = generate_mask(data, index, fig)
        file_prog.close()
        sys.stderr = std_err_backup

        # Visibility
        fig.data[0].visible = True
        fig.data[len(fig.data)//2].visible = True

        steps = []
        for i in range(len(fig.data)//2):
            step = dict(
                method="restyle",
                args=[{"visible": [False] * len(fig.data)}],
            )
            # Toggle i'th trace to "visible"
            step["args"][0]["visible"][i] = True
            # Toggle i'th trace to "visible"
            step["args"][0]["visible"][i+len(fig.data)//2] = True
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Dicom file: "},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
        )

    fig.update_layout(template="plotly_dark")
    fig.update_layout(paper_bgcolor='#1e1e1e',
                      plot_bgcolor='#1e1e1e')
    return fig


def update_graph_3d():
    """Draw prediction graphes."""
    fig = go.Figure()
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # Helix equation
    t = np.linspace(0, 20, 100)
    x, y, z = np.cos(t), np.sin(t), t

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(template="plotly_dark")
    fig.update_layout(paper_bgcolor='#1e1e1e',
                      plot_bgcolor='#1e1e1e')
    return fig


def parse_contents(contents, filenames, dates):
    """Parse content."""
    response = None
    if not filenames:
        return response

    if len(filenames) > 1:
        data = []
        files_types = []
        for f, c in zip(filenames, contents):
            if f.endswith('.dcm') or f.endswith('.txt'):
                data.append(c)
                files_types.append(f)
        if data:
            # Should send back dict with same element has the
            # predictions.pbz2
            config = fs.get_configs_root() / 'web_config.yml'
            response = infer.get_inference(data,
                                           files_types,
                                           str(config))
    else:
        contents = contents[0]
        filenames = filenames[0]
        content_type, content_string = contents.split(',')
        if filenames.endswith('.pbz2'):
            decoded = base64.b64decode(content_string)
            decoded = io.BytesIO(decoded)
            with bz2.BZ2File(decoded, 'rb') as f:
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
            out += float(val)*0
        elif key == "sex":
            out += -float(val)*(0.1)
        elif key == "hypertension":
            out += float(val)*(0)
        elif key == "renal":
            out += float(val)*(0.5)
        elif key == "implantation":
            out += float(val)*(0.25)
        elif key == "cholesterol":
            out += float(val)*(0.04)
        elif key == "ldl":
            out += float(val)*(0.07)
        elif key == "gradient":
            out += float(val)*(0.002)
        elif key == "size":
            out += -float(val)*(0.01)
        elif key == "score":
            out += float(val)*(0.08)

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
            out += float(val)*0
        elif key == "sex":
            out += -float(val)*(0.2)
        elif key == "hypertension":
            out += float(val)*(0)
        elif key == "renal":
            out += float(val)*(0.5)
        elif key == "implantation":
            out += float(val)*(0.28)
        elif key == "cholesterol":
            out += float(val)*(0.03)
        elif key == "ldl":
            out += float(val)*(0.02)
        elif key == "gradient":
            out += float(val)*(0.5)
        elif key == "size":
            out += -float(val)*(0.01)
        elif key == "score":
            out += float(val)*(0.15)

    return int(np.round(out, 0))


def operations(coeff_dict):
    """Estimate operation score."""
    output_1 = two_years_model(coeff_dict)
    output_2 = five_years_model(coeff_dict)
    if isinstance(output_1, int) \
       and isinstance(output_2, int):
        return [output_1, output_2]
    else:
        message = 'Missing informations for Clinical Outcomes'
        return message
