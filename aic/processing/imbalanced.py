#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for processing medical data inputs
and data for training/infering.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

LABEL_CHANNELS = {
    "labels": {
        "background": 0,
        "other": 1,
        "Magna_valve": 2,
    }
}


def imbalanced_data_counter(image, msks):
    """Deal with imbalanced data.

    Get a repartition of the ratio of the different classes.
    Go through the dataset.json file.
    This done image wise and pixel wise
    """
    # Pixel Wise
    total_pixel = (
        image.shape[0] * image.shape[1] * image.shape[2] * image.shape[3]
    )

    print("\n")
    for key, value in LABEL_CHANNELS["labels"].items():
        count = (msks[:, :, :, 0] == value).sum()
        ratio = 100 * count / total_pixel
        print("pixel wise ratio (%) of {} is {}".format(key, str(ratio)))

    # Image Wise
    for key, value in LABEL_CHANNELS["labels"].items():
        count = 0
        for index in range(msks.shape[0]):
            if value == 0:
                is_value = np.all((msks[index, :, :, 0] == value))
            else:
                is_value = np.any((msks[index, :, :, 0] == value))
            if is_value:
                count += 1
        print(
            "image wise ratio (%) of {} is {}".format(
                key, str(count / msks.shape[0])
            )
        )
    print("\n")


def imbalanced_data_augmentation(imgs, msks, total=20, seed=42):
    """Augment data.

    Construct the image generator for data augmentation then
    initialize the total number of images generated thus far.
    """
    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )

    msks_stack = []
    for i in tqdm(range(msks.shape[0])):
        msks_ = msks[i, :, :, :]
        extra_channel = np.zeros((msks.shape[1], msks.shape[2], 1))
        msks_ = np.concatenate((msks_, extra_channel), axis=2)
        msks_ = np.expand_dims(msks_, 0)
        # prepare iterator
        it = aug.flow(msks_, batch_size=1, seed=seed)
        # generate samples
        for i in range(total):
            batch = it.next()
            msks_stack.append(batch)

    imgs_stack = []
    for i in tqdm(range(imgs.shape[0])):
        imgs_ = np.expand_dims(imgs[i, :, :, :], 0)
        # prepare iterator
        it = aug.flow(imgs_, batch_size=1, seed=seed)
        # generate samples
        for i in range(total):
            batch = it.next()
            imgs_stack.append(batch)

    imgs_augmented = np.vstack(imgs_stack)
    msks_augmented = np.vstack(msks_stack)
    msks_augmented = msks_augmented[:, :, :, :2]

    return imgs_augmented, msks_augmented
