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
import aic.model.loaders as ld
ld.load_framework(sys)


if __name__ == '__main__':

    config = ld.load_config('./data_info.yml')
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

    model = ld.load_model(model_name, model_version)
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
