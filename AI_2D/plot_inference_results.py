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
import sys 
import yaml
from aic_models.dataloader import DatasetGenerator,get_filelist,slice_filelist
import psutil
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

from aic_models.model import unet

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

def dice_coef(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    prediction = K.backend.round(prediction)  # Round to 0 or 1

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)

def soft_dice_coef(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson (Soft) Dice  - Don't round the predictions
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)

def tversky(self, target, prediction, smooth=1, alpha=0.7):

    # Flatten the input data
    if self.channels_first:
        y_true = K.backend.permute_dimensions(target, (3,1,2,0))
        y_pred = K.backend.permute_dimensions(prediction, (3,1,2,0))
    else :
        y_true = target
        y_pred = prediction

    y_true_pos = K.backend.flatten(y_true)
    y_pred_pos = K.backend.flatten(y_pred)
    true_pos = K.backend.sum(y_true_pos * y_pred_pos)
    false_neg = K.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.backend.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                (1 - alpha) * false_pos + smooth)

def plot_results(imgs,labels,model,folder,number):
    """
    Calculate the Dice and plot the predicted masks for image # img_no
    """

    # Prediction
    if model is not None:
        predictions = model.predict(imgs)
    
    for i in range(imgs.shape[0]):
        # Init Figure
        plt.figure(figsize=(20,20))

        # Image 
        plt.subplot(1,3,1)
        plt.imshow(imgs[i, :, :, 0], cmap="bone", origin="lower")
        plt.title("MRI")
        plt.axis("off")

        # Label
        plt.subplot(1, 3, 2)
        plt.imshow(labels[i, :, :, 0],origin="lower",vmin=0, vmax=1)
        plt.title("Ground Truth")
        plt.axis("off")

        if model is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(predictions[i, :, :, 0],origin="lower",vmin=0, vmax=1)
            plt.title("Predictions\n(Dice {:.4f}, Soft Dice {:.4f})".\
                      format(dice_coef(labels[i],predictions[i]),soft_dice_coef(labels[i],predictions[i])))
            #plt.title("Predictions\n(Tversky {:.4f})".\
            #           format(dice_coef(labels[i],predictions[i])))
            plt.axis("off")
        
        png_filename = os.path.join(folder, "pred_{}.png".format(number+i))
        plt.savefig(png_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":

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

    #model_filename = "/home/francoismasson/Project_AIC/Viewer/models/unet_model_for_aic_512.hdf5"
    model_filename = "/home/francoismasson/Project_AIC/output/unet_model_for_aic_test.hdf5"

    """
    Load a model, load the data, and see inference.
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
    ds_train = DatasetGenerator(trainFiles,trainLabels,num_slices_per_scan,batch_size=batch_size,\
                                crop_dim=[crop_dim,crop_dim], augment=True,imbalanced=True)
    ds_validation = DatasetGenerator(validateFiles,validateLabels,num_slices_per_scan,batch_size=batch_size,\
                                crop_dim=[crop_dim,crop_dim], augment=False)
    ds_test = DatasetGenerator(testFiles,testLabels,num_slices_per_scan,batch_size=batch_size,\
                                crop_dim=[crop_dim,crop_dim], augment=False)
     
    unet_model = unet()
    try :
        model = unet_model.load_model(model_filename)
    except :
        model = None
    #model = None

    # Create output directory for images
    png_directory = "inference_examples"

    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    png_folder = os.path.join(Path.cwd(),png_directory)

    # The plots will be saved to the png_directory (Keep only the first batch for now)
    number = 0
    loader = ds_test
    for img,label in loader.ds:
        plot_results(img,label,model,png_folder,number)
        number += img.shape[0]