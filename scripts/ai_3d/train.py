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

This script train a 3D Unet model from a TensorFlow/Keras model.
using a DatasetGenerator architecture, and then saves the
best model.
"""


import datetime
import yaml
import os
import aic.misc.files as fs
import aic.misc.utils as ut
from aic.model.architecture.model_3D import unet
from aic.processing.dataloader import DatasetGenerator3D
from aic.misc.setting_tf import requirements_3d as req3d

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

    data_path = config.get("data_path", None)
    batch_size = config.get("batch_size", None)
    crop_dim = config.get("crop_dim", None)
    if crop_dim:
        crop_dim = tuple(crop_dim)
    channels_first = config.get("channels_first", None)
    filters = config.get("filters", None)
    learning_rate = config.get("learning_rate", None)
    weight_dice_loss = config.get("weight_dice_loss", None)
    output_path = config.get("output_path", None)
    inference_filename = config.get("inference_filename", None)
    if inference_filename:
        inference_filename += \
            datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + \
            '.hdf5'
    use_upsampling = config.get("use_upsampling", None)
    print_model = config.get("print_model", None)
    z_slice_min = config.get("z_slice_min", None)
    z_slice_max = config.get("z_slice_max", None)
    epochs = config.get("epochs", None)
    json_filename = config.get("json_filename", None)
    json_filename = os.path.join(data_path, json_filename)
    blocktime, num_inter_threads, num_threads = req3d()

    """
    Create a model, load the data, and train it.
    """

    """
    Step 1: Define a data loader
    """
    print("-" * 30)
    print("Loading the data from the Valve project directory" +
          "to a TensorFlow data loader ...")
    print("-" * 30)

    data = DatasetGenerator3D(crop_dim=crop_dim,
                              data_path=data_path,
                              json_filename=json_filename,
                              batch_size=batch_size,
                              number_output_classes=1,
                              random_seed=816
                              )

    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """

    unet_model = unet(channels_first=channels_first,
                      filters=filters,
                      use_upsampling=use_upsampling,
                      learning_rate=learning_rate,
                      weight_dice_loss=weight_dice_loss,
                      output_path=output_path,
                      inference_filename=inference_filename,
                      blocktime=blocktime,
                      num_threads=num_threads,
                      num_inter_threads=num_inter_threads,
                      print_model=print_model)

    model = unet_model.create_model(crop_dim,
                                    crop_dim)
    model_filename, model_callbacks = unet_model.get_callbacks()

    """
    Step 3: Train the model on the data
    """
    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    steps_per_epoch = data.num_files // batch_size
    model.fit(data.get_train(),
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=data.get_validate(),
              verbose=1,
              callbacks=model_callbacks)

    """
    Step 4: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet_model.evaluate_model(model_filename, data.get_test())

    print("Total time elapsed for program = {} seconds".format(
          datetime.datetime.now() - START_TIME))
    print("Stopped script on {}".format(datetime.datetime.now()))
