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
Training of a 2D Unet model.

This script train a 2D Unet model from a TensorFlow/Keras model.
using a DatasetGenerator architecture, and then saves the
best model.
"""


import datetime
import yaml
import os
import aic.misc.files as fs
import aic.misc.utils as ut
from aic.model.architecture.model_2D import Unet
from aic.processing.dataloader import DatasetGenerator2D
from aic.misc.setting_tf import requirements_2d as req2d

if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    print("Started script on {}".format(START_TIME))

    """
    Load the config required for the model
    """
    config = str(fs.get_configs_root() / "train_config_2d.yml")
    with open(config) as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config.get("data_path", None)
    batch_size = config.get("batch_size", None)
    crop_dim = config.get("crop_dim", None)
    channels_first = config.get("channels_first", None)
    featuremaps = config.get("featuremaps", None)
    output_path = config.get("output_path", None)
    inference_filename = config.get("inference_filename", None)
    if inference_filename:
        inference_filename += (
            datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".hdf5"
        )
    use_dropout = config.get("use_dropout", None)
    use_upsampling = config.get("use_upsampling", None)
    learning_rate = config.get("learning_rate", None)
    weight_dice_loss = config.get("weight_dice_loss", None)
    print_model = config.get("print_model", None)
    z_slice_min = config.get("z_slice_min", None)
    z_slice_max = config.get("z_slice_max", None)
    epochs = config.get("epochs", None)
    json_filename = config.get("json_filename", None)
    json_filename = os.path.join(data_path, json_filename)
    blocktime, num_inter_threads, num_threads = req2d()

    """
    Create a model, load the data, and train it.
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

    # This is the maximum value one of the files haves.
    # Required because model built with assumption all file same z slice.
    # Imbalanced True --> reduce number of only 0 layers
    num_slices_per_scan = ut.slice_file_list(
        data_path=data_path, json_filename=json_filename
    )
    ds_train = DatasetGenerator2D(
        trainFiles,
        trainLabels,
        num_slices_per_scan,
        batch_size=batch_size,
        crop_dim=[crop_dim, crop_dim],
        augment=True,
        imbalanced=True,
        z_slice_min=-1,
        z_slice_max=-1,
    )
    ds_validation = DatasetGenerator2D(
        validateFiles,
        validateLabels,
        num_slices_per_scan,
        batch_size=batch_size,
        crop_dim=[crop_dim, crop_dim],
        augment=False,
        imbalanced=False,
        z_slice_min=z_slice_min,
        z_slice_max=z_slice_max,
    )
    ds_test = DatasetGenerator2D(
        testFiles,
        testLabels,
        num_slices_per_scan,
        batch_size=batch_size,
        crop_dim=[crop_dim, crop_dim],
        augment=False,
        z_slice_min=z_slice_min,
        z_slice_max=z_slice_max,
    )

    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """

    unet_model = Unet(
        channels_first=channels_first,
        fms=featuremaps,
        output_path=output_path,
        inference_filename=inference_filename,
        learning_rate=learning_rate,
        weight_dice_loss=weight_dice_loss,
        use_upsampling=use_upsampling,
        use_dropout=use_dropout,
        print_model=print_model,
        blocktime=blocktime,
        num_threads=num_threads,
        num_inter_threads=num_inter_threads,
    )

    model = unet_model.create_model(
        ds_train.get_input_shape(), ds_train.get_output_shape()
    )
    model_filename, model_callbacks = unet_model.get_callbacks()

    """
    Step 3: Train the model on the data
    """
    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_validation,
        verbose=1,
        callbacks=model_callbacks,
    )

    """
    Step 4: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet_model.evaluate_model(model_filename, ds_test)

    print(
        "Total time elapsed for program = {} seconds".format(
            datetime.datetime.now() - START_TIME
        )
    )
    print("Stopped script on {}".format(datetime.datetime.now()))
