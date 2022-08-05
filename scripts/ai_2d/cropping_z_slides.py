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

This module loads the data from data.py, creates a TensorFlow/Keras model
from model.py, trains the model on the data, and then saves the
best model.
"""


import yaml
import os
import aic.misc.files as fs
import aic.misc.utils as ut
from tqdm import tqdm
import numpy as np
from aic.misc.setting_tf import requirements_2d as req2d


if __name__ == "__main__":

    """
    Load the config required for the model
    """
    config = str(fs.get_configs_root() / "train_config_2d.yml")
    with open(config) as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config.get("data_path", None)
    json_filename = config.get("json_filename", None)
    json_filename = os.path.join(data_path, json_filename)
    blocktime, num_inter_threads, num_threads = req2d()

    """
    Estimate the ratio of slices that can be cropped
    to deal with imbalanced dataset.
    """

    print("-" * 30)
    print(
        "Loading the data from the Valve project directory"
        + "to a TensorFlow data loader ..."
    )
    print("-" * 30)

    files = ut.get_file_list(
        data_path=data_path, json_filename=json_filename, split=1.0
    )
    imgs = files[0]
    labels = files[1]

    ratio_label = {}
    ratio_min = []
    ratio_max = []
    for label_filename in tqdm(labels):
        index = []
        label = ut.load_mask(label_filename)
        # Stack the loaded npy files
        label = [np.load(label[i]) for i in range(len(label))]
        label = np.stack(label, axis=0)
        label[label == 1] = 0.0
        label[label == 2] = 1.0
        for z in range(label.shape[0]):
            # Check if all 2D numpy array contains only 0
            result = np.all((label[z] == 0))
            if not result:
                index.append(z)
        index = np.array(index)
        ratio_label[label_filename] = index
        ratio_min.append(index[0] / label.shape[0])
        ratio_max.append(index[-1] / label.shape[0])

    print(min(ratio_min))
    print(max(ratio_max))
