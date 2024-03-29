#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Inference Web based for the model prediction
"""

import base64
import io
import os
import shutil
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from vedo.io import load as vedo_load
from vedo.volume import Volume
from vtk.util.numpy_support import vtk_to_numpy

import aic.misc.utils as ut
import aic.model.from_tflite as tfl
import aic.model.loaders as ld
import aic.processing.fitting as ft
import aic.processing.operations as op
import aic.processing.preprocess as dp
import aic.processing.scoring as sc


def get_inference(
    data,
    file_types=None,
    config="./model_info.yml",
    model=None,
    online=True,
    lite=True,
    save_path="./cache",
):
    """Get inference results."""
    if not model:
        return {}
    slices = []
    if file_types:
        for f, d in zip(file_types, data):
            content_type, content_string = d.split(",")
            if f.endswith(".txt"):
                patient_info = content_string.encode("utf8")
                patient_info = base64.decodebytes(patient_info).decode("UTF-8")
            elif f.endswith(".dcm"):
                decoded = base64.b64decode(content_string)
                slices.append(io.BytesIO(decoded))
    else:
        slices = [data]

    if slices:
        if os.path.exists(config):
            config = ld.load_config(config)
            model_version = config.get("model_version", None)
            crop_dim = config.get("crop_dim", -1)
            z_slice_min = config.get("z_slice_min", None)
            z_slice_max = config.get("z_slice_max", None)
            threshold = config.get("threshold", None)
        # Get Dicom files
        path = "./cache/tmp/"
        if not online:
            list_data = os.listdir(data)
            list_data = [
                os.path.join(data, d) for d in list_data if d.endswith(".dcm")
            ]
        else:
            list_data = deepcopy(slices)
        dataset = ut.get_slices(list_data)
        image = op.get_pixels_hu(dataset)
        if not online:
            if os.path.exists(path) and os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            for name in list_data:
                shutil.copy(name, path)
        else:
            ut.save_dicom(slices, path=path)
        img = vedo_load(path)
        spacing = img.imagedata().GetSpacing()
        area = spacing[0] * spacing[1]
        dimensions = img.imagedata().GetDimensions()
        # Get a vtk volume object
        predictions_agatston = Volume(img.imagedata())
        # Get the all points in isosurface
        iso = img.isosurface()
        points = iso.GetMapper().GetInput()
        all_array = points.GetPoints()
        all_numpy_nodes = vtk_to_numpy(all_array.GetData())
        # x y z center from isovolume
        center = np.array(
            [
                (np.min(all_numpy_nodes[:, 0]) + np.max(all_numpy_nodes[:, 0]))
                / 2,
                (np.min(all_numpy_nodes[:, 1]) + np.max(all_numpy_nodes[:, 1]))
                / 2,
                (np.min(all_numpy_nodes[:, 2]) + np.max(all_numpy_nodes[:, 2]))
                / 2,
            ]
        )
        # Get predictions
        predictions, crop_values = get_predictions(
            model,
            model_version,
            data=path,
            crop_dim=crop_dim,
            z_slice_max=z_slice_max,
            z_slice_min=z_slice_min,
            spacing=spacing,
            dimensions=dimensions,
            lite=lite,
        )
        vertices, _ = op.make_mesh(predictions, -1)
        # Clustering
        vertices_predictions = op.clustering(
            vertices,
            model_version,
            center,
            all_numpy_nodes,
            ratio=0.4,
            threshold=6000,
            max_=None,
            dimensions=dimensions,
            spacings=spacing,
            crop_values=crop_values,
        )
        # Get encapsulated Volume
        predictions_agatston_points = op.boxe_3d(
            predictions_agatston, vertices_predictions
        )
        # Get the all points in isosurface Mesh/Volume
        predictions_agatston_points = op.to_points(predictions_agatston)
        predictions_final_points_threshold = predictions_agatston_points[
            predictions_agatston_points[:, 3] > threshold
        ]
        # Convex-Hull estimation
        hull = ft.convex_hull(predictions_final_points_threshold[:, :3])
        mask = op.isInHull(predictions_agatston_points[:, :3], hull)
        predictions_agatston_points = predictions_agatston_points[mask]
        w_fit, C_fit, r_fit, fit_err = ft.fitting_cylinder(
            predictions_final_points_threshold[:, :3], guess_angles=None
        )
        mask_agatston, valve, candidate = op.get_candidates(
            predictions_agatston_points,
            w_fit=w_fit,
            r_fit=r_fit,
            threshold=threshold,
            spacing=spacing,
            dimensions=dimensions,
        )
        score = sc.agatston_score(
            image, mask_agatston, area, threshold_min=130, threshold_max=None
        )
        results = sc.save_prediction(
            image,
            mask_agatston,
            path=save_path,
            score=score,
            area=area,
            threshold_min=130,
            threshold_max=None,
            valve=valve,
            candidate=candidate,
            online=online,
        )
        if not online:
            shutil.rmtree(path)
        return results


def get_predictions(
    model,
    model_version=1,
    data=None,
    crop_dim=-1,
    z_slice_max=None,
    z_slice_min=None,
    spacing=None,
    dimensions=None,
    lite=False,
):
    """Get model predictions."""
    if isinstance(data, str):
        idx = os.path.join(data)
        img = ut.load_scan(idx)
    else:
        img = data
    img = op.get_pixels_hu(img)
    crop_values = None
    # old version, input images were normalized for each slice
    if model_version == 0:
        img = dp.preprocess_inputs(img)
    # new version, input images were normalized according to z
    elif model_version == 1:
        # padding
        # Need for the mesh reconstruct
        padding = np.zeros(img.shape) - 2
        if crop_dim != -1:
            img = dp.crop_dim_2d(img, crop_dim=crop_dim)
        if (z_slice_min is not None) and (z_slice_max is not None):
            min_ = int(z_slice_min * img.shape[0])
            max_ = int(z_slice_max * img.shape[0])
            index_z_crop = np.arange(min_, max_)
            img = img[index_z_crop]
        img = dp.preprocess_img(img)
        img = np.expand_dims(img, -1)
    else:
        print("Unknown/Unsupported version")
        exit()

    pred_list = []
    # https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
    for i in tqdm(range(img.shape[0])):
        pred = np.expand_dims(img[i, :, :, :], 0)
        if not lite:
            prediction = model.predict(pred, verbose=0)
        else:
            prediction = tfl.get_interpreter(model, input=pred)
        if model_version == 0:
            prediction = np.argmax(prediction.squeeze(), axis=-1)
            prediction = np.rot90(prediction, axes=(1, 0))
            prediction = np.expand_dims(prediction, 0)
            prediction[prediction == 0] = -1
        elif model_version == 1:
            prediction = prediction[0, :, :, 0]
            prediction = np.rot90(prediction, axes=(1, 0))
            prediction = np.expand_dims(prediction, 0)
            # Binary Classification
            prediction[prediction > 0.5] = 1
            prediction[prediction < 0.5] = 0
            # Set -2 to allow marching cube algo reconstruction of prediction
            prediction[prediction != 1] = -2
        else:
            print("Unknown/Unsupported version")
            exit()
        pred_list.append(prediction)
    predictions = np.vstack(pred_list)

    # Padding reconstruction
    if model_version == 1:
        if crop_dim != -1:
            xc = (dimensions[0] - crop_dim) // 2
            yc = (dimensions[1] - crop_dim) // 2
        else:
            xc = 0
            yc = 0
        if (z_slice_min is not None) and (z_slice_max is not None):
            padding[
                index_z_crop, xc : xc + img.shape[1], yc : yc + img.shape[2]
            ] = predictions
        else:
            padding[
                :, xc : xc + img.shape[1], yc : yc + img.shape[2]
            ] = predictions
        predictions = padding
        crop_values = [
            xc * spacing[0],
            xc + img.shape[1] * spacing[0],
            yc * spacing[1],
            yc + img.shape[2] * spacing[1],
        ]

    predictions, _ = op.resample(
        predictions, [spacing[0], spacing[1]], spacing[2], [1, 1, 1]
    )
    return predictions, crop_values
