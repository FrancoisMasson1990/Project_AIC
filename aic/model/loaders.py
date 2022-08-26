#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Config loader.
"""

import os

import yaml

import aic.misc.files as fs


def load_model(model_name, model_version=1):
    """Load AI model."""
    model = None
    if model_name is not None:
        # model_2D_old is a deprecated model generated with tf1 version
        if not os.path.exists(model_name):
            model_name = fs.get_models_root() / model_name
        else:
            print(f"Could not find {model_name}")
            model_name = None
        if model_name:
            if model_version == 0:
                from aic.model.architecture import model_2D_old

                unet_model = model_2D_old.Unet()
                model = unet_model.load_model(model_name, False)
            elif model_version == 1:
                from aic.model.architecture import model_2D

                unet_model = model_2D.Unet()
                model = unet_model.load_model(model_name)
            print("-" * 30)
            print("Model load successfully")
            print("-" * 30)
    return model


def load_tflitemodel(model_name):
    """Load TFLite AI model."""
    model = None
    if model_name is not None:
        if not os.path.exists(model_name):
            model_name = fs.get_models_root() / model_name
        else:
            print(f"Could not find {model_name}")
            model_name = None
        if model_name:
            import tflite_runtime.interpreter as tflite

            model = tflite.Interpreter(model_path=str(model_name))
            print("-" * 30)
            print("Model load successfully")
            print("-" * 30)
    return model


def load_config(path):
    """Load a config file."""
    config = {}
    if path.endswith(".yml"):
        with open(path) as f:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            config = yaml.load(f, Loader=yaml.FullLoader)
    return config
