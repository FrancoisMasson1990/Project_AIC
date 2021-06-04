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
import sys 
import yaml
from aic_models.model import unet
from aic_models.dataloader import DatasetGenerator,get_filelist,slice_filelist
import psutil

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

    START_TIME = datetime.datetime.now()
    print("Started script on {}".format(START_TIME))

    """
    Load the config required for the model
    """

    with open('./train_config.yml') as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config.get("data_path",None)
    batch_size = config.get("batch_size",None)
    crop_dim = config.get("crop_dim",None)
    channels_first = config.get("channels_first",None)
    featuremaps = config.get("featuremaps",None)
    output_path = config.get("output_path",None)
    inference_filename = config.get("inference_filename",None)
    use_dropout = config.get("use_dropout",None)
    use_upsampling = config.get("use_upsampling",None)
    learning_rate = config.get("learning_rate",None)
    weight_dice_loss = config.get("weight_dice_loss",None)
    print_model = config.get("print_model",None)

    epochs = config.get("epochs",None)
    
    """
    Create a model, load the data, and train it.
    """

    """
    Step 1: Define a data loader
    """
    print("-" * 30)
    print("Loading the data from the Valve project directory to a TensorFlow data loader ...")
    print("-" * 30)

    trainFiles,trainLabels,validateFiles,validateLabels,testFiles,testLabels = get_filelist(data_path=data_path)
    
    # This is the maximum value one of the files haves. 
    # Required because model built with assumption all file same z slice.
    num_slices_per_scan = slice_filelist(data_path=data_path)
    ds_train = DatasetGenerator(trainFiles,trainLabels,num_slices_per_scan,batch_size=batch_size, crop_dim=[crop_dim,crop_dim], augment=True)
    ds_validation = DatasetGenerator(validateFiles,validateLabels,num_slices_per_scan,batch_size=batch_size, crop_dim=[crop_dim,crop_dim], augment=False)
    ds_test = DatasetGenerator(testFiles,testLabels,num_slices_per_scan,batch_size=batch_size, crop_dim=[crop_dim,crop_dim], augment=False)
  
    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """

    unet_model = unet(channels_first=channels_first,
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
                      num_inter_threads=num_inter_threads)

    model = unet_model.create_model(ds_train.get_input_shape(), ds_train.get_output_shape())
    model_filename, model_callbacks = unet_model.get_callbacks()

    """
    Step 3: Train the model on the data
    """
    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    model.fit(ds_train,
              epochs=epochs,
              validation_data=ds_validation,
              verbose=1,
              callbacks=model_callbacks)

    exit()
    """
    Step 4: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet_model.evaluate_model(model_filename, ds_test)

    print("Total time elapsed for program = {} seconds".format(
          datetime.datetime.now() - START_TIME))
    print("Stopped script on {}".format(datetime.datetime.now()))