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
from miscnn_dependency import *  # noqa: E402


if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    print("Started script on {}".format(START_TIME))

    """
    Load the config required for the model
    """
    config = str(fs.get_configs_root() / "train_config_3d.yml")
    with open(config) as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)
    blocktime, num_inter_threads, num_threads = req3d()

    data_path = config.get("data_path", fs.get_dataset_root())
    interface = NIFTI_interface(pattern="AIC-00[0-9]*", channels=1, classes=2)
    # Initialize data path and create the Data I/O instance
    data_io = Data_IO(interface, data_path)

    # Create and configure the Data Augmentation class
    data_aug = Data_Augmentation(
        cycles=2,
        scaling=True,
        rotations=True,
        elastic_deform=True,
        mirror=True,
        brightness=True,
        contrast=True,
        gamma=True,
        gaussian_noise=True,
    )

    # Create a pixel value normalization Subfunction through Z-Score
    sf_normalize = Normalization(mode="z-score")
    # Create a clipping Subfunction
    sf_clipping = Clipping(min=0, max=1000)
    # Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
    sf_resample = Resampling((3.22, 1.62, 1.62))

    # Assemble Subfunction classes into a list
    # Be aware that the Subfunctions will be exectue
    # according to the list order!
    subfunctions = [sf_resample, sf_clipping, sf_normalize]

    # Create a Preprocessor instance to configure
    # how to preprocess the data into batches
    pp = Preprocessor(
        data_io,
        data_aug=data_aug,
        batch_size=4,
        subfunctions=subfunctions,
        prepare_subfunctions=True,
        analysis="patchwise-crop",
        patch_shape=(32, 64, 64),
        use_multiprocessing=True,
    )

    # Create a deep learning neural network model with
    # a standard U-Net architecture
    unet_standard = Architecture()
    model = miscnn.Neural_Network(preprocessor=pp, architecture=unet_standard)
