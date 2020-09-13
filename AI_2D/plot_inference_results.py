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
Takes a trained model and performs inference on a few validation examples.
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras as K 
import h5py

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
# Load model
from model import unet  
from pathlib import Path 
import yaml

LABEL_CHANNELS = {"labels":{
	 			  "background":0,
				  "other":1,
	 			  "Magna_valve":2,
				 }}


def calc_dice(target, prediction, smooth=0.01):
    """
    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth (target) mask and P is the prediction mask
    """
    prediction = np.round(prediction)

    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


def calc_soft_dice(target, prediction, smooth=0.01):
    """
    Sorensen (Soft) Dice coefficient - Don't round preictions
    """
    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


def plot_results(model, imgs_validation, msks_validation,
                 img_no,png_directory):
    """
    Calculate the Dice and plot the predicted masks for image # img_no
    """

    img = imgs_validation[[img_no], ]
    msk = msks_validation[[img_no], ]

    if model is not None:
        pred_mask = model.predict(img)

        if not intel_model:
            msk_plot = msk.squeeze()
            msk_plot = np.argmax(msk_plot,axis=-1)
            pred_plot = pred_mask.squeeze()
            pred_plot = np.argmax(pred_plot,axis=-1)
        
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img[0, :, :, 0], cmap="bone", origin="lower")
    plt.title("MRI")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    if intel_model :
        plt.imshow(msk[0, :, :, 0],origin="lower",vmin=0, vmax=1)
        plt.title("Ground Truth")
    else :
        plt.imshow(msk_plot,origin="lower")
    plt.title("Ground Truth")
    plt.axis("off")   
    plt.subplot(1, 3, 3)
    if model is not None:
       if intel_model :
           plt.imshow(pred_mask[0, :, :, 0],origin="lower",vmin=0, vmax=1)
       else :
           plt.imshow(pred_plot,origin="lower",vmin=0, vmax=1)
       plt.title("Prediction\n(Dice = {:.4f})".format(calc_dice(msk, pred_mask)))
    plt.axis("off")

    png_filename = os.path.join(png_directory, "pred_{}.png".format(img_no))
    plt.savefig(png_filename, bbox_inches="tight", pad_inches=0)
    if model is not None :
        print("Dice {:.4f}, Soft Dice {:.4f}, Saved png file to: {}".format(
            calc_dice(msk, pred_mask), calc_soft_dice(msk, pred_mask), png_filename))

if __name__ == "__main__":

    data_filename = "/home/francoismasson/Desktop/Project_AIC/hdf5_files/project_aic.h5"
    model_filename = "/home/francoismasson/Desktop/Project_AIC/output/unet_model_for_aic.hdf5"

    with open('./train_config.yml') as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    intel_model = config.get("intel_model",None)
    
    # Load data
    df = h5py.File(data_filename, "r")
    imgs = df["imgs_testing"]
    msks = df["msks_testing"]
 
    unet_model = unet()
    try :
        model = unet_model.load_model(model_filename,intel_model)
    except :
        model = None

    # Create output directory for images
    png_directory = "inference_examples"

    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    png_folder = os.path.join(Path.cwd(),png_directory)

    # Plot some results
    # The plots will be saved to the png_directory
    indicies_testing = np.arange(0,imgs.shape[0]).tolist()

    for idx in indicies_testing:
        plot_results(model, imgs, msks, idx, png_folder)