#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Print out a visual representation of the inference from a 3D trained model.
"""

import os
from pathlib import Path

import yaml
from tqdm import tqdm

import aic.misc.files as fs
import aic.misc.plots as plt
import aic.misc.utils as ut
from aic.misc.setting_tf import requirements_3d as req3d
from aic.model.architecture import model_3D

if __name__ == "__main__":

    """
    Load the config required for the model
    """
    config = str(fs.get_configs_root() / "train_config_3d.yml")
    with open(config) as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config.get("data_path", None)
    crop_dim = config.get("crop_dim", -1)
    if isinstance(crop_dim, list):
        crop_dim = tuple(crop_dim)
    resize_dim = config.get("resize_dim", -1)
    batch_size = config.get("batch_size", None)
    json_filename = config.get("json_filename", None)
    json_filename = os.path.join(data_path, json_filename)
    blocktime, num_inter_threads, num_threads = req3d()

    # model_version = 1
    model_filename = str(fs.get_models_root() / "unet_3d_model_for_aic.hdf5")

    """
    Load a model, load the data, and see inference.
    """

    """
    Step 1: Define a data loader
    """
    print("-" * 30)
    print(
        "Loading the data from the Valve project directory"
        + "to a TensorFlow data loader ..."
    )
    print("-" * 30)

    files = ut.get_file_list(data_path=data_path, json_filename=json_filename)
    trainFiles = files[0]
    trainLabels = files[1]
    validateFiles = files[2]
    validateLabels = files[3]
    testFiles = files[4]
    testLabels = files[5]

    unet_model = model_3D.Unet()
    # model_filename = None
    if model_filename is not None:
        model = unet_model.load_model(model_filename)
    else:
        model = None

    # Create output directory for images
    png_directory = "inference_examples"
    png_folder = os.path.join(Path.cwd(), png_directory)

    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # The plots will be saved to the png_directory
    imgs = trainFiles
    labels = trainLabels
    for index, (img, label) in tqdm(
        enumerate(zip(imgs, labels)), total=len(imgs)
    ):
        valve_name = str(Path(img).parent).split("/")[-1]
        plt.plot_results_3d(
            imgs=img,
            labels=label,
            model=model,
            crop_dim=crop_dim,
            resize_dim=resize_dim,
            folder=png_folder,
            name=valve_name,
            randomize=False,
        )
