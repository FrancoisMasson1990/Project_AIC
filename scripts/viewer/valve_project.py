#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Main viewer allowing to visualize data, infere model
and make predictions.
"""

import os
import sys
import aic.viewer.viewer_2D as v2d
import aic.viewer.viewer_3D as v3d
import aic.misc.files as fs
import yaml
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus and len(sys.argv) > 1 and sys.argv[1].startswith("-a"):
    print("allowing growth")
    growth = True
else:
    print("nogrowth")
    growth = False
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, growth)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
    print(e)


if __name__ == '__main__':

    with open('./data_info.yml') as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config.get("data_path", fs.get_valve_root())
    labels_2D = config.get("labels_2D", None)
    multi_label = config.get("multi_label", None)
    model_name = config.get("model_name", None)
    model_version = config.get("model_version", None)
    template = config.get("template", None)
    crop_dim = config.get("crop_dim", -1)
    z_slice_min = config.get("z_slice_min", None)
    z_slice_max = config.get("z_slice_max", None)
    threshold = config.get("threshold", None)
    spacing = config.get("spacing", None)
    kwargs = {"crop_dim": crop_dim,
              "z_slice_min": z_slice_min,
              "z_slice_max": z_slice_max,
              "threshold": threshold,
              "spacing": spacing}

    if model_name is not None:
        # model_2D_old is a deprecated model generated with tf1 version
        if not os.path.exists(model_name):
            model_name = fs.get_models_root() / model_name
        else:
            print(f"Could not find {model_name}")
            model = None
        if model_version == 0:
            from aic.model import model_2D_old
            unet_model = model_2D_old.unet()
            model = unet_model.load_model(model_name,
                                          False)
        elif model_version == 1:
            from aic.model import model_2D
            unet_model = model_2D.unet()
            model = unet_model.load_model(model_name)
        print("-" * 30)
        print("Model load successfully")
        print("-" * 30)
    else:
        model = None

    surface_label_path = os.path.join(data_path,
                                      "labels_2d_npy")
    volume_label_path = os.path.join(data_path,
                                     "labels_3d_npy")
    data_path = os.path.join(data_path,
                             "datasets_dcm")
    sub_folders = os.listdir(data_path)
    data = []
    for sub_folder in sub_folders:
        root = os.path.join(data_path,
                            sub_folder)
        sub_ = os.listdir(root)
        for sub in sub_:
            data.append(os.path.join(root,
                                     sub))

    if labels_2D:
        v2d.Viewer2D(data_path=data,
                     folder_mask=surface_label_path)
    else:
        viewer = \
            v3d.Viewer3D(data_path=data,
                         mode=4,
                         label=surface_label_path,
                         npy=volume_label_path,
                         multi_label=multi_label,
                         model=model,
                         template=template,
                         model_version=model_version,
                         **kwargs)
        viewer.show()
