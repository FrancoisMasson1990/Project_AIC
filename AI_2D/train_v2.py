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
import multiprocessing
import psutil
import datetime
import os
import tensorflow as tf
from tensorflow import keras as K 
import yaml
from data import load_data
import numpy as np
from model import unet

"""
For best CPU speed set the number of intra and inter threads
to take advantage of multi-core systems.
See https://github.com/intel/mkl-dnn
"""

#CONFIG = tf.ConfigProto(intra_op_parallelism_threads=args.num_threads,
#                        inter_op_parallelism_threads=args.num_inter_threads)

#SESS = tf.Session(config=CONFIG)
#SESS = tf.Session()
#K.backend.set_session(SESS)

print("TensorFlow version: {}".format(tf.__version__))
print("Keras API version: {}".format(K.__version__))

def train_and_predict(hdf5_filename = None,
                      output_path = None,
                      inference_filename = None,
                      batch_size = None,
                      n_epoch = None,
                      crop_dim = None,
                      use_augmentation = None,
                      channels_first = None,
                      seed = None,
                      featuremaps = None,
                      blocktime = None,
                      num_threads = None,
                      learning_rate = None,
                      weight_dice_loss = None,
                      num_inter_threads = None,
                      use_upsampling = None,
                      use_dropout = None,
                      print_model = None):
    """
    Create a model, load the data, and train it.
    """

    """
    Step 1: Load the data
    """

    print("-" * 30)
    print("Loading the data from HDF5 file ...")
    print("-" * 30)

    imgs_train, msks_train, imgs_validation, msks_validation, imgs_testing, msks_testing = \
        load_data(hdf5_filename,batch_size,[crop_dim,crop_dim],use_augmentation,channels_first,seed)

    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """
    
    unet_model = unet(channels_first = channels_first,
                      fms = featuremaps,
                      output_path = output_path,
                      inference_filename = inference_filename,
                      batch_size = batch_size,
                      blocktime = blocktime,
                      num_threads = num_threads,
                      learning_rate = learning_rate,
                      weight_dice_loss = weight_dice_loss,
                      num_inter_threads = num_inter_threads,
                      use_upsampling = use_upsampling,
                      use_dropout = use_dropout,
                      print_model = print_model)
    
    model = unet_model.create_model(imgs_train.shape, msks_train.shape)
    model_filename, model_callbacks = unet_model.get_callbacks()

    # If there is a current saved file, then load weights and start from
    # there.
    saved_model = os.path.join(output_path,inference_filename)
    if os.path.isfile(saved_model):
        model.load_weights(saved_model)

    """
    Step 3: Train the model on the data
    """
    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    model.fit(imgs_train, msks_train,
              batch_size=batch_size,
              epochs=n_epoch,
              validation_data=(imgs_validation, msks_validation),
              verbose=1, shuffle="batch",
              callbacks=model_callbacks)

    """
    Step 4: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet_model.evaluate_model(model_filename, imgs_testing, msks_testing)

    """
    Step 5: Save frozen TensorFlow version of model
    This can be convert into OpenVINO format with model optimizer.
    """
    print("-" * 30)
    print("Freezing model and saved to a TensorFlow protobuf ...")
    print("-" * 30)
    unet_model.save_frozen_model(model_filename, imgs_testing.shape)

if __name__ == "__main__":

    with open('./train_config.yml') as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    start_time = datetime.datetime.now()
    print("Started script on {}".format(start_time))

    data_filename = config.get("data_filename",None)
    output_path = config.get("output_path",None)
    inference_filename = config.get("inference_filename",None)
    batch_size = config.get("batch_size",None)
    n_epoch = config.get("n_epoch",None)
    crop_dim = config.get("crop_dim",None)
    use_augmentation = config.get("use_augmentation",None)
    channels_first = config.get("channels_first",None)
    seed = config.get("seed",None)
    featuremaps = config.get("featuremaps",None)
    blocktime = config.get("blocktime",None)
    num_threads = min(len(psutil.Process().cpu_affinity()), psutil.cpu_count(logical=False))
    learning_rate = config.get("learning_rate",None)
    weight_dice_loss = config.get("weight_dice_loss",None)
    num_inter_threads = config.get("num_inter_threads",None)
    use_upsampling = config.get("use_upsampling",None)
    use_dropout = config.get("use_dropout",None)
    print_model = config.get("print_model",None)
    

    # Set environment 

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["KMP_BLOCKTIME"] = str(blocktime)
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["INTER_THREADS"] = str(num_inter_threads)
    os.environ["INTRA_THREADS"] = str(num_threads)
    os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime


    train_and_predict(hdf5_filename = data_filename,
                      output_path = output_path,
                      inference_filename = inference_filename,
                      batch_size = batch_size,
                      n_epoch = n_epoch,
                      crop_dim = crop_dim,
                      use_augmentation = use_augmentation,
                      channels_first = channels_first,
                      seed = seed,
                      featuremaps = featuremaps,
                      blocktime = blocktime,
                      num_threads = num_threads,
                      learning_rate = learning_rate,
                      weight_dice_loss = weight_dice_loss,
                      num_inter_threads = num_inter_threads,
                      use_upsampling = use_upsampling,
                      use_dropout = use_dropout,
                      print_model = print_model)

    print("Total time elapsed for program = {} seconds".format(datetime.datetime.now() - start_time))
    print("Stopped script on {}".format(datetime.datetime.now()))
