#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library to show inference results.
"""

import numpy as np
import os
import aic.misc.utils as ut
from aic.processing import metrics as mt
from aic.processing import preprocess as dp
from aic.processing import operations as op
import matplotlib.pyplot as plt


def plot_results(imgs,
                 labels,
                 model,
                 crop_dim,
                 z_slice_min,
                 z_slice_max,
                 folder,
                 name,
                 model_version):
    """Plot the predicted masks for image."""
    # Image processing
    imgs = ut.load_scan(imgs)
    imgs = op.get_pixels_hu(imgs)

    if crop_dim != -1:
        imgs = dp.crop_dim(imgs, crop_dim=crop_dim)

    min_ = int(z_slice_min*imgs.shape[0])
    max_ = int(z_slice_max*imgs.shape[0])
    index_z_crop = np.arange(min_, max_)
    imgs = imgs[index_z_crop]

    # old version, input images were normalized for each slice
    if model_version == 0:
        imgs = dp.preprocess_inputs(imgs)
    # new version, input images were normalized according to z
    elif model_version == 1:
        imgs = dp.preprocess_img(imgs)
        imgs = np.expand_dims(imgs, -1)

    # label processing
    labels = ut.load_mask(labels)
    labels = dp.preprocess_label(labels)
    if crop_dim != -1:
        labels = dp.crop_dim(labels, crop_dim=crop_dim)

    labels = labels[index_z_crop]

    # Init Figure
    if model is None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        ax0, ax1 = ax[0], ax[1]
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 15))
        ax0, ax1, ax2 = ax[0], ax[1], ax[2]

    ax0_object = None
    ax1_object = None
    ax2_object = None
    for i in range(imgs.shape[0]):

        # Prediction
        if model is not None:
            prediction = np.expand_dims(imgs[i, :, :, :], 0)
            prediction = model.predict(prediction)

        # Image
        if ax0_object is None:
            ax0_object = ax0.imshow(
                imgs[i, :, :, 0], cmap="bone", origin="lower")
        else:
            ax0_object.set_data(imgs[i, :, :, 0])

        ax0.set_title("MRI")
        ax0.axis("off")

        # Label
        if ax1_object is None:
            ax1_object = ax1.imshow(
                labels[i, :, :], origin="lower", vmin=0, vmax=1)
        else:
            ax1_object.set_data(labels[i, :, :])

        ax1.set_title("Ground Truth")
        ax1.axis("off")

        if model is not None:
            if ax2_object is None:
                ax2_object = ax2.imshow(
                    prediction[0, :, :, 0], origin="lower", vmin=0, vmax=1)
            else:
                ax2_object.set_data(prediction[0, :, :, 0])

            ax2.set_title("Predictions\n(Dice {:.4f}, Soft Dice {:.4f})".
                          format(
                              mt.dice_coef(
                                  np.expand_dims(
                                      labels[i, :, :],
                                      (0, -1)),
                                  prediction),
                              mt.soft_dice_coef(
                                  np.expand_dims(
                                         labels[i, :, :],
                                         (0, -1)),
                                  prediction)))
            ax2.axis("off")

        png_filename = os.path.join(folder, "pred_{}_{}.png".format(name, i))
        plt.savefig(png_filename, bbox_inches="tight", pad_inches=0)