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

from natsort import natsorted

import aic.misc.files as fs
import aic.model.loaders as ld
import aic.viewer.viewer_2D as v2d
import aic.viewer.viewer_3D as v3d
from aic.misc.setting_tf import requirements_2d as req2d

if __name__ == "__main__":
    # Set TensorFlow requirements
    req2d()

    config = fs.get_configs_root() / "viewer_config.yml"
    config = ld.load_config(str(config))

    data_path = config.get("data_path", fs.get_valve_root())
    labels_2D = config.get("labels_2D", None)
    config.pop("data_path", None)
    config.pop("labels_2D", None)
    surface_label_path = os.path.join(data_path, "labels_2d_npy")
    volume_label_path = os.path.join(data_path, "labels_3d_npy")
    data_path = os.path.join(data_path, "datasets_dcm")
    sub_folders = os.listdir(data_path)
    data = []
    include_list = []
    for sub_folder in natsorted(sub_folders):
        sampled = False
        if not include_list:
            sampled = True
        if sub_folder in include_list:
            sampled = True
        if sampled:
            root = os.path.join(data_path, sub_folder)
            sub_ = os.listdir(root)
            for sub in sub_:
                if os.path.isdir(os.path.join(root, sub)):
                    data.append(os.path.join(root, sub))

    if labels_2D:
        v2d.Viewer2D(data_path=data, folder_mask=surface_label_path)
    else:
        viewer = v3d.Viewer3D(
            data_path=data,
            mode=4,
            label=surface_label_path,
            npy=volume_label_path,
            **config,
        )
        viewer.show()
