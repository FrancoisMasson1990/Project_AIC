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

import os
import numpy as np
import base64
import io
import aic.processing.preprocess as dp
import aic.misc.utils as ut
import aic.processing.operations as op
import aic.misc.files as fs
from tqdm import tqdm
import yaml


def get_inference(data,
                  file_types,
                  config="./model_info.yml"
                  ):
    # Perform model prediction using a specific model
    # Box fitting
    # Cylinder fitting
    # Score estimation
    # Prediction
    # Should provide yaml config instead of all the arguments

    if os.path.exists(config):
        with open(config) as f:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            config = yaml.load(f, Loader=yaml.FullLoader)
            model_name = config.get("model_name", None)
            model_version = config.get("model_version", None)
            crop_dim = config.get("crop_dim", -1)
            z_slice_min = config.get("z_slice_min", None)
            z_slice_max = config.get("z_slice_max", None)

    if model_name is not None:
        # model_2D_old is a deprecated model generated with tf1 version
        if not os.path.exists(model_name):
            model_name = fs.get_models_root() / model_name
            from aic.model import model_2D
            unet_model = model_2D.unet()
            model = unet_model.load_model(model_name)
        else:
            print(f"Could not find {model_name}")
            model = None

    slices = []
    for f, d in zip(file_types, data):
        content_type, content_string = d.split(',')
        if f.endswith('.txt'):
            patient_info = \
                content_string.encode("utf8")
            patient_info = \
                base64.decodebytes(patient_info).decode('UTF-8')
        elif f.endswith('.dcm'):
            decoded = base64.b64decode(content_string)
            slices.append(io.BytesIO(decoded))

    if slices:
        img = ut.get_slices(slices)
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
                img = \
                    dp.crop_dim(img,
                                crop_dim=crop_dim)
            if (z_slice_min is not None) \
                    and (z_slice_max is not None):
                min_ = int(z_slice_min*img.shape[0])
                max_ = int(z_slice_max*img.shape[0])
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
            prediction = model.predict(pred)
            if model_version == 0:
                prediction = np.argmax(prediction.squeeze(), axis=-1)
                prediction = np.rot90(prediction, axes=(1, 0))
                prediction = np.expand_dims(prediction, 0)
                prediction[prediction == 0] = -1
            elif model_version == 1:
                prediction = prediction[0, :, :, 0]
                prediction = np.rot90(prediction, axes=(1, 0))
                prediction = np.expand_dims(prediction, 0)
                prediction[prediction != 1.0] = -2
            else:
                print("Unknown/Unsupported version")
                exit()
            pred_list.append(prediction)
        predictions = np.vstack(pred_list)
        print(predictions.shape)
    return None


def load_config():
    pass


def load_model():
    pass
