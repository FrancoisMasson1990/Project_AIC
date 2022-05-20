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

import psutil
import os
import tensorflow as tf
import sys
import yaml
from tqdm import tqdm
import numpy as np
from aic.misc.utils import get_file_list, load_mask
import psutil


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
# If hyperthreading is enabled, then use
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
# If hyperthreading is NOT enabled, then use
# os.environ["KMP_AFFINITY"] = "granularity=thread,compact"

blocktime = 0
num_inter_threads = 1
num_threads = min(len(psutil.Process().cpu_affinity()),
                  psutil.cpu_count(logical=False))

os.environ["KMP_BLOCKTIME"] = str(blocktime)

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["INTRA_THREADS"] = str(num_threads)
os.environ["INTER_THREADS"] = str(num_inter_threads)
os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus and len(sys.argv) > 1:
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

# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/

if __name__ == "__main__":

    """
    Load the config required for the model
    """

    with open('./train_config.yml') as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config.get("data_path", None)

    """
    Estimate the ratio of cropping in x_y
    that can be cropped to deal with dataset
    """

    print("-" * 30)
    print("Loading the data from the Valve project directory" +
          "to a TensorFlow data loader ...")
    print("-" * 30)

    imgs, labels, _, _, _, _ = get_file_list(data_path=data_path, split=1.0)

    min_x = []
    max_x = []
    min_y = []
    max_y = []
    for label_filename in tqdm(labels):
        index = []
        label = load_mask(label_filename)
        # Stack the loaded npy files
        label = [np.load(label[i]) for i in range(len(label))]
        label = np.stack(label, axis=0)
        label[label == 1] = 0.0
        label[label == 2] = 1.0
        for z in range(label.shape[0]):
            # Check where values are not 0
            result = np.argwhere(label[z] != 0)
            if len(result) > 0:
                index.append(result)
        index = np.concatenate(index)
        min_x.append(np.min(index[:, 0]))
        max_x.append(np.max(index[:, 0]))
        min_y.append(np.min(index[:, 1]))
        max_y.append(np.max(index[:, 1]))

    print(min(min_x))
    print(max(max_x))
    print(min(min_y))
    print(max(max_y))
