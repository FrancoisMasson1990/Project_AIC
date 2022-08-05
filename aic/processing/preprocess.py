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
from scipy import ndimage
from skimage.transform import resize

LABEL_CHANNELS = {
    "labels": {
        "background": 0,
        "other": 1,
        "Magna_valve": 2,
    }
}


def normalize_img(img):
    """Normalize the pixel values.

    This is one of the most important preprocessing steps.
    We need to make sure that the pixel values have a mean of 0
    and a standard deviation of 1 to help the model to train
    faster and more accurately.
    """
    for channel in range(img.shape[-1]):
        img[:, :, :, channel] = (
            img[:, :, :, channel] - np.mean(img[:, :, :, channel])
        ) / np.std(img[:, :, :, channel])

    return img


def crop_center(img, cropx, cropy, cropz):
    """Take a center crop of the images.

    If we are using a 2D model, then we'll just stack the
    z dimension.
    """
    z, x, y, c = img.shape

    # Make sure starting index is >= 0
    startx = max(x // 2 - (cropx // 2), 0)
    starty = max(y // 2 - (cropy // 2), 0)

    # Make sure ending index is <= size
    endx = min(startx + cropx, x)
    endy = min(starty + cropy, y)

    return img[:, startx:endx, starty:endy, :]


def resampling(img, size):
    """Resampling images."""
    new_image = np.empty((img.shape[0], size, size, 1))
    for i in range(img.shape[0]):
        new_image[i] = resize(img[i], (size, size))
    return new_image


def preprocess_inputs(img, resize=-1):
    """Process the input images."""
    if len(img.shape) != 4:  # Make sure 4D
        img = np.expand_dims(img, -1)

    if resize != -1:
        img = crop_center(img, resize, resize, -1)

    img = normalize_img(img)

    return img


def preprocess_label_v1(msk, intel_model=False, resize=-1):
    """Process the ground truth labels."""
    # Stack the loaded npy files
    msk = [np.load(msk[i]) for i in range(len(msk))]
    msk = np.stack(msk, axis=0)

    if intel_model:
        if len(msk.shape) != 4:  # Make sure 4D
            msk = np.expand_dims(msk, -1)
    else:
        # extract certain classes from mask
        msks = [(msk == v) for v in LABEL_CHANNELS["labels"].values()]
        msk = np.stack(msks, axis=-1).astype("float")

    # Cropping
    if resize != -1:
        msk = crop_center(msk, resize, resize, -1)

    # WIP : Trying to find labels with no data imbalanced
    # Remove one label
    # msk = np.delete(msk,1,3) #Removed Others

    index = []
    for index in range(msk.shape[0]):
        is_value = np.all((msk[index, :, :, 1] == 0))
        if not is_value:
            index.append(index)

    return msk, np.array(index)


def preprocess_label(label):
    """Set label attribution.

    Please refer LABEL_CHANNEL for the mask attribution.
    """
    # Stack the loaded npy files
    label = [np.load(label[i]) for i in range(len(label))]
    label = np.stack(label, axis=0)
    # Took the decision to set to 0 other labels and to 1 magna valve
    label[label == 1] = 0.0
    label[label == 2] = 1.0

    return label


def preprocess_label_3d(label, resize_dim, number_output_classes=1):
    """Set label attribution.

    Please refer LABEL_CHANNEL for the mask attribution.
    """
    # Stack the loaded npy files
    label = [np.load(label[i]) for i in range(len(label))]
    label = np.stack(label, axis=0)
    # Took the decision to set to 0 other labels and to 1 magna valve
    # label[label == 1] = 0.0
    # label[label == 2] = 1.0
    label = np.moveaxis(label, 0, -1)
    if resize_dim != -1:
        label = resize_input(
            label, width=resize_dim[0], height=resize_dim[1], depth=resize_dim[2]
        )
    # Combine all masks but background
    if number_output_classes == 1:
        label[label > 0] = 1.0
        label = np.expand_dims(label, -1)
    else:
        label_temp = np.zeros(list(label.shape) + [number_output_classes])
        for channel in range(number_output_classes):
            label_temp[label == channel, channel] = 1.0
        label = label_temp
    return label


def preprocess_img(img):
    """Preprocess images.

    Preprocessing for the image
    z-score normalize
    """
    # Based on vtk algorithm :
    # scrange -> img.GetScalarRange() [min,max values]
    # threshold = (2 * scrange[0] + scrange[1]) / 3.0
    # 500 or 1000 is a good threshold based on observation for Magna valve
    img[img < 0] = 0
    # Read Intensity normalization in medical images from
    # https://theaisummer.com/medical-image-processing/
    # Scale applied plays a crucial role in training
    img[img > 1000] = 1000
    return (img - img.mean()) / img.std()


def preprocess_img_3d(img, resize_dim, min_=0, max_=1000):
    """Preprocess images.

    Preprocessing for the image
    z-score normalize
    """
    img[img < min_] = min_
    # Read Intensity normalization in medical images from
    # https://theaisummer.com/medical-image-processing/
    # Scale applied plays a crucial role in training
    img[img > max_] = max_
    img = (img - min_) / (max_ - min_)
    img = img.astype("float32")
    img = np.moveaxis(img, 0, -1)
    if resize_dim != -1:
        img = resize_input(
            img, width=resize_dim[0], height=resize_dim[1], depth=resize_dim[2]
        )
    img = np.expand_dims(img, -1)
    return img


def crop_dim_2d(img, crop_dim):
    """Crop images.

    Crop around the center of the images based on size provided
    If we are using a 2D model, then we'll just stack the
    z dimension.
    """
    if img.ndim == 3:
        z, x, y = img.shape
    elif img.ndim == 4:
        z, x, y, c = img.shape

    # Make sure starting index is >= 0
    startx = max(x // 2 - (crop_dim // 2), 0)
    starty = max(y // 2 - (crop_dim // 2), 0)

    # Make sure ending index is <= size
    endx = min(startx + crop_dim, x)
    endy = min(starty + crop_dim, y)

    if img.ndim == 3:
        return img[:, startx:endx, starty:endy]
    elif img.ndim == 4:
        return img[:, startx:endx, starty:endy, :]


def crop_dim_3d(img, msk, crop_dim, randomize):
    """Randomly crop the image and mask."""
    slices = []
    # Do we randomize?
    is_random = randomize and np.random.rand() > 0.5
    for idx in range(len(img.shape) - 1):  # Go through each dimension

        cropLen = crop_dim[idx]
        imgLen = img.shape[idx]

        start = (imgLen - cropLen) // 2

        ratio_crop = 0.20  # Crop up this % of pixels for offset
        # Number of pixels to offset crop in this dimension
        offset = int(np.floor(start * ratio_crop))

        if offset > 0:
            if is_random:
                start += np.random.choice(range(-offset, offset))
                # Don't fall off the image
                if (start + cropLen) > imgLen:
                    start = (imgLen - cropLen) // 2
        else:
            start = 0

        slices.append(slice(start, start + cropLen))

    return img[tuple(slices)], msk[tuple(slices)]


def augment_data_3d(img, msk, crop_dim):
    """Get Data augmentation.

    Flip image and mask. Rotate image and mask.
    """
    # Determine if axes are equal and can be rotated
    # If the axes aren't equal then we can't rotate them.
    equal_dim_axis = []
    for idx in range(0, len(crop_dim)):
        for jdx in range(idx + 1, len(crop_dim)):
            if crop_dim[idx] == crop_dim[jdx]:
                equal_dim_axis.append([idx, jdx])  # Valid rotation axes
    dim_to_rotate = equal_dim_axis

    if np.random.rand() > 0.5:
        # Random 0,1 (axes to flip)
        ax = np.random.choice(np.arange(len(crop_dim) - 1))
        img = np.flip(img, ax)
        msk = np.flip(msk, ax)

    elif (len(dim_to_rotate) > 0) and (np.random.rand() > 0.5):
        rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        # This will choose the axes to rotate
        # Axes must be equal in size
        random_axis = dim_to_rotate[np.random.choice(len(dim_to_rotate))]
        img = np.rot90(img, rot, axes=random_axis)  # Rotate axes 0 and 1
        msk = np.rot90(msk, rot, axes=random_axis)  # Rotate axes 0 and 1
    return img, msk


def resize_input(input_, width=128, height=128, depth=64):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = depth
    desired_width = width
    desired_height = height

    # Get current depth
    current_depth = input_.shape[-1]
    current_width = input_.shape[0]
    current_height = input_.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    input_ = ndimage.interpolation.zoom(
        input_, (width_factor, height_factor, depth_factor), order=1
    )
    return input_
