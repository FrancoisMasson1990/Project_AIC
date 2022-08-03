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
Training of a 3D Unet model.

This script use MisCnn library to train a 3D Unet model
from a TensorFlow/Keras model.
"""

import datetime
import warnings
import yaml
import aic.misc.files as fs
from aic.misc.setting_tf import requirements_3d as req3d
warnings.filterwarnings("ignore")
import miscnn  # noqa: E402
from miscnn.data_loading.interfaces import (
    NIFTI_interface)  # noqa: E402
from miscnn.neural_network.architecture.unet.standard import (
    Architecture)  # noqa: E402


if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    print("Started script on {}".format(START_TIME))

    """
    Load the config required for the model
    """
    config = str(fs.get_configs_root() / 'train_config_3d.yml')
    with open(config) as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)
    blocktime, num_inter_threads, num_threads = req3d()
    data_path = config.get("data_path", fs.get_dataset_root())
    interface = NIFTI_interface(pattern="AIC-00[0-9]*",
                                channels=1, classes=3)
    # Initialize data path and create the Data I/O instance
    data_io = miscnn.Data_IO(interface, data_path)

    # Create a Preprocessor instance to configure
    # how to preprocess the data into batches
    pp = miscnn.Preprocessor(data_io, batch_size=4,
                             analysis="patchwise-crop",
                             patch_shape=(64, 64, 64),
                             use_multiprocessing=True)

    # Create a deep learning neural network model with
    # a standard U-Net architecture

    unet_standard = Architecture()
    model = miscnn.Neural_Network(preprocessor=pp,
                                  architecture=unet_standard)
