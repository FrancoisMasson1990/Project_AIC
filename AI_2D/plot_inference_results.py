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

import numpy as np
from aic_models import data_preprocess as dp
import psutil
import os
import tensorflow as tf
from tensorflow import keras as K
import sys 
import yaml
from aic_models.dataloader import DatasetGenerator,get_file_list,slice_file_list
import psutil
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from aic_models.model_2D import unet

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

    prediction = K.backend.round(prediction)  # Round to 0 or 1
    
    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)

def tversky(self, target, prediction, smooth=1, alpha=0.7,channels_first=False):

    # Flatten the input data
    if channels_first:
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

def plot_results(imgs,labels,model,folder,name,model_version):
    """
    Calculate the Dice and plot the predicted masks for image # img_no
    """

    # Image processing
    imgs = dp.load_scan(imgs)
    imgs = dp.get_pixels_hu(imgs)

    if crop_dim != -1:
        imgs = dp.crop_dim(imgs,crop_dim=crop_dim)

    min_ = int(z_slice_min*imgs.shape[0])
    max_ = int(z_slice_max*imgs.shape[0])
    index_z_crop = np.arange(min_,max_)
    imgs = imgs[index_z_crop]

    if model_version == 0:  # old version, input images were normalized for each slice
        imgs = dp.preprocess_inputs(img)
    elif model_version == 1: # new version, input images were normalized according to z
        imgs = dp.preprocess_img(imgs)
        imgs = np.expand_dims(imgs,-1)

    # label processing
    labels = dp.load_mask(labels)
    labels = dp.preprocess_label(labels)
    if crop_dim != -1:
        labels = dp.crop_dim(labels,crop_dim=crop_dim)

    labels = labels[index_z_crop]

    # Init Figure
    if model is None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        ax0, ax1 = ax[0], ax[1]
    else :
        fig, ax = plt.subplots(1, 3, figsize=(15, 15))
        ax0, ax1, ax2 = ax[0], ax[1], ax[2]

    ax0_object = None
    ax1_object = None
    ax2_object = None
    for i in range(imgs.shape[0]):
        
        # Prediction
        if model is not None:
           prediction = np.expand_dims(imgs[i,:,:,:], 0)
           prediction = model.predict(prediction)

        # Image 
        if ax0_object is None:
            ax0_object = ax0.imshow(imgs[i, :, :, 0], cmap="bone", origin="lower")
        else :
            ax0_object.set_data(imgs[i, :, :, 0])

        ax0.set_title("MRI")
        ax0.axis("off")

        # Label
        if ax1_object is None:
            ax1_object = ax1.imshow(labels[i, :, :],origin="lower",vmin=0, vmax=1)
        else :
            ax1_object.set_data(labels[i, :, :])

        ax1.set_title("Ground Truth")
        ax1.axis("off")

        if model is not None:
            if ax2_object is None:
                ax2_object = ax2.imshow(prediction[0, :, :, 0],origin="lower",vmin=0, vmax=1)
            else :
                ax2_object.set_data(prediction[0, :, :, 0])
            
            ax2.set_title("Predictions\n(Dice {:.4f}, Soft Dice {:.4f})".\
                            format(dice_coef(np.expand_dims(labels[i,:,:],(0,-1)),prediction),
                            soft_dice_coef(np.expand_dims(labels[i,:,:],(0,-1)),prediction)))
            #ax2.set_title("Predictions\n(Tversky {:.4f})".\
            #          format(dice_coef(np.expand_dims(labels[i,:,:],(0,-1)),prediction)))
            ax2.axis("off")
        
        png_filename = os.path.join(folder, "pred_{}_{}.png".format(name,i))
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
    crop_dim = config.get("crop_dim",-1)
    channels_first = config.get("channels_first",None)
    featuremaps = config.get("featuremaps",None)
    output_path = config.get("output_path",None)
    inference_filename = config.get("inference_filename",None)
    z_slice_min = config.get("z_slice_min",None)
    z_slice_max = config.get("z_slice_max",None)

    model_version = 1
    model_filename = "/home/francoismasson/Project_AIC/Viewer/models/unet_model_for_aic_320.hdf5"

    """
    Load a model, load the data, and see inference.
    """

    """
    Step 1: Define a data loader
    """
    print("-" * 30)
    print("Loading the data from the Valve project directory to a TensorFlow data loader ...")
    print("-" * 30)

    trainFiles,trainLabels,validateFiles,validateLabels,testFiles,testLabels = get_file_list(data_path=data_path)
     
    unet_model = unet()
    #model_filename = None
    if model_filename is not None:
        if model_version == 0:
            model = unet_model.load_model(model_filename,False)
        elif model_version == 1:
            model = unet_model.load_model(model_filename)
    else :
        model = None

    # Create output directory for images
    png_directory = "inference_examples"

    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    png_folder = os.path.join(Path.cwd(),png_directory)

    # The plots will be saved to the png_directory
    imgs = trainFiles
    labels = trainLabels

    for index,(img,label) in tqdm(enumerate(zip(imgs,labels)),total=len(imgs)): 
        valve_name = str(Path(img).parent).split('/')[-1]
        plot_results(img,label,model,png_folder,valve_name,model_version)