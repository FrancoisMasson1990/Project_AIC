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
from aic_models.dataloader import get_file_list
import psutil
from aic_models import data_preprocess as dp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
# If hyperthreading is enabled, then use
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
# If hyperthreading is NOT enabled, then use
#os.environ["KMP_AFFINITY"] = "granularity=thread,compact"

blocktime = 0
num_inter_threads = 1
num_threads = min(len(psutil.Process().cpu_affinity()), psutil.cpu_count(logical=False))

os.environ["KMP_BLOCKTIME"] = str(blocktime)

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["INTRA_THREADS"] = str(num_threads)
os.environ["INTER_THREADS"] = str(num_inter_threads)
os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus and len(sys.argv)> 1:
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

    data_path = config.get("data_path",None)
    
    """
    Estimate the ratio of slices that can be cropped to deal with imbalanced dataset
    """

    print("-" * 30)
    print("Loading the data from the Valve project directory to a TensorFlow data loader ...")
    print("-" * 30)

    imgs,labels,_,_,_,_ = get_file_list(data_path=data_path,split=1.0)

    ratio_label = {}
    ratio_min = []
    ratio_max = []
    for label_filename in tqdm(labels):
        index = []
        label = dp.load_mask(label_filename)
        ## Stack the loaded npy files
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
        ratio_min.append(index[0]/label.shape[0])
        ratio_max.append(index[-1]/label.shape[0])

    print(min(ratio_min))
    print(max(ratio_max))