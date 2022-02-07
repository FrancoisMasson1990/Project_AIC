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
import datetime
import os
import tensorflow as tf
import sys 
import yaml
from aic_models.model_3D import unet
# From the 2D
#from aic_models.dataloader import DatasetGenerator,
from aic_models.dataloader import get_file_list,slice_file_list
#from the 3D
from dataloader import DataGenerator
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
    z_slice_min = config.get("z_slice_min",None)
    z_slice_max = config.get("z_slice_max",None)

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

    trainFiles,trainLabels,validateFiles,validateLabels,testFiles,testLabels = get_file_list(data_path=data_path)
    # ds_train = DataGenerator("train",     # ["train", "validate", "test"]
    #                          data_path,    # File path for data
    #                          train_test_split=0.85,  # Train test split
    #                          validate_test_split=0.5,  # Validation/test split
    #                          batch_size=8,  # batch size
    #                          dim=(128, 128, 128),  # Dimension of images/masks
    #                          n_in_channels=1,  # Number of channels in image
    #                          n_out_channels=1,  # Number of channels in mask
    #                          shuffle=True,  # Shuffle list after each epoch
    #                          augment=False,   # Augment images
    #                          seed=816)
    
    # This is the maximum value one of the files haves. 
    # Required because model built with assumption all file same z slice.
    # Imbalanced True --> reduce number of only 0 layers
    # num_slices_per_scan = slice_file_list(data_path=data_path)
    # ds_train = DatasetGenerator(trainFiles,trainLabels,num_slices_per_scan,batch_size=batch_size,\
    #                             crop_dim=[crop_dim,crop_dim], augment=True,imbalanced=True, z_slice_min=-1, z_slice_max=-1)
    # ds_validation = DatasetGenerator(validateFiles,validateLabels,num_slices_per_scan,batch_size=batch_size,\
    #                             crop_dim=[crop_dim,crop_dim], augment=False, imbalanced=False, z_slice_min=z_slice_min, z_slice_max=z_slice_max)
    # ds_test = DatasetGenerator(testFiles,testLabels,num_slices_per_scan,batch_size=batch_size,\
    #                             crop_dim=[crop_dim,crop_dim], augment=False, z_slice_min=z_slice_min, z_slice_max=z_slice_max)
       
    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """
    
    unet_model = unet(use_upsampling=False, 
                      learning_rate=0.001,
                      n_cl_in=1, 
                      n_cl_out=1, 
                      feature_maps = 16,
                      dropout=0.2, print_summary=True,
                      channels_last = True)
    

    model = unet_model.create_model()
    model_filename, model_callbacks = unet_model.get_callbacks()

    exit()
    
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