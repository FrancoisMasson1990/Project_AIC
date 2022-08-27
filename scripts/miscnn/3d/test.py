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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

import aic.misc.files as fs
from aic.misc.setting_tf import requirements_3d as req3d

warnings.filterwarnings("ignore")
import dependency.miscnn_dependency as deps  # noqa: E402

if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    print("Started script on {}".format(START_TIME))

    """
    Load the config required for the model
    """
    config = str(fs.get_configs_root() / "train_config_3d_miscnn.yml")
    with open(config) as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)
    blocktime, num_inter_threads, num_threads = req3d()

    data_path = "../data_kidney/"

    interface = deps.NIFTI_interface(
        pattern="case_000[0-9]*", channels=1, classes=3
    )
    # Initialize data path and create the Data I/O instance
    data_io = deps.Data_IO(interface, data_path)

    # Create and configure the Data Augmentation class
    data_aug = deps.Data_Augmentation(
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
    sf_normalize = deps.Normalization(mode="z-score")
    # Create a clipping Subfunction
    sf_clipping = deps.Clipping(min=0, max=1000)
    # Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
    sf_resample = deps.Resampling((3.22, 1.62, 1.62))

    # Assemble Subfunction classes into a list
    # Be aware that the Subfunctions will be exectue
    # according to the list order!
    subfunctions = [sf_resample, sf_clipping, sf_normalize]

    # Create a Preprocessor instance to configure
    # how to preprocess the data into batches
    pp = deps.Preprocessor(
        data_io,
        data_aug=data_aug,
        batch_size=1,
        subfunctions=subfunctions,
        prepare_subfunctions=True,
        analysis="patchwise-crop",
        patch_shape=(64, 128, 128),
        use_multiprocessing=True,
    )

    # Adjust the patch overlap for predictions
    # pp.patchwise_overlap = (40, 80, 80)

    # Create a deep learning neural network model with
    # a standard U-Net architecture
    unet_standard = deps.Architecture()
    model = deps.Neural_Network(
        preprocessor=pp,
        loss=deps.tversky_loss,
        metrics=[deps.dice_soft, deps.dice_crossentropy],
        architecture=unet_standard,
        batch_queue_size=2,
        workers=8,
    )
    cb_ckpt = ModelCheckpoint(
        filepath="./model", verbose=1, monitor="loss", save_best_only=True
    )
    cb_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=20,
        verbose=1,
        mode="min",
        min_delta=0.0001,
        cooldown=1,
        min_lr=0.00001,
    )
    cb_es = EarlyStopping(
        monitor="loss", min_delta=0, patience=150, verbose=1, mode="min"
    )
    sample_list = data_io.get_indiceslist()
    model.train(sample_list, epochs=250, callbacks=[cb_ckpt, cb_lr, cb_es])
    model.predict(sample_list[0])
