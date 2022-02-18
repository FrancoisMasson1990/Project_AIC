#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Various of functions simplifies execution.
"""

import numpy as np
import os
import json
from natsort import natsorted
import glob
import pydicom


def get_file_list(data_path, seed=816, split=0.7):
    """Get list of files.

    Shuffling of the dataset required for the model training/evaluation
    """
    experiment_data = json_export(data_path)

    # Print information about the Magna valve experiment data
    print("*" * 30)
    print("=" * 30)
    print("Dataset name:        ", experiment_data["name"])
    print("Dataset description: ", experiment_data["description"])
    print("Tensor image size:   ", experiment_data["tensorImageSize"])
    print("Dataset release:     ", experiment_data["release"])
    print("=" * 30)
    print("*" * 30)

    """
    Randomize the file list. Then separate into training and
    validation lists. We won't use the testing set since we
    don't have ground truth masks for this; instead we'll
    split the validation set into separate test and validation
    sets.
    """

    image_files = expand_list(experiment_data["dataset_folder"])
    label_files = expand_list(experiment_data["label_folder"])

    assert len(image_files) == len(
        label_files), "Files and labels don't have the same length"

    # Set the random seed so that always get same random mix
    np.random.seed(seed)
    numFiles = len(image_files)
    idxList = np.arange(numFiles)  # List of file indices
    # Shuffle the indices to randomize train/test/split
    np.random.shuffle(idxList)

    # index for the end of the training files
    trainIdx = int(np.floor(numFiles*split))
    trainList = idxList[:trainIdx]

    otherList = idxList[trainIdx:]
    numOther = len(otherList)
    otherIdx = numOther//2  # index for the end of the testing files
    validateList = otherList[:otherIdx]
    testList = otherList[otherIdx:]

    trainList = [0]
    validateList = [1]
    testList = [2]
    # trainList = validateList = testList

    trainFiles = []
    trainLabels = []
    for idx in trainList:
        trainFiles.append(image_files[idx])
        trainLabels.append(label_files[idx])

    validateFiles = []
    validateLabels = []
    for idx in validateList:
        validateFiles.append(image_files[idx])
        validateLabels.append(label_files[idx])

    testFiles = []
    testLabels = []
    for idx in testList:
        testFiles.append(image_files[idx])
        testLabels.append(label_files[idx])

    print("Number of training files   = {}".format(len(trainList)))
    print("Number of validation files = {}".format(len(validateList)))
    print("Number of testing files    = {}".format(len(testList)))

    return trainFiles, trainLabels, validateFiles, \
        validateLabels, testFiles, testLabels


def slice_file_list(data_path):
    """Get the max number of slice in the dataset.

    Required for the Dataloader Generator
    """
    experiment_data = json_export(data_path)
    num_slice_max = expand_list(
        experiment_data["dataset_folder"], format='/*.dcm')
    num_slice_max = max(len(x) for x in num_slice_max)

    return num_slice_max


def json_export(data_path):
    """Extract dataset informations contained into a json file."""
    json_filename = os.path.join(data_path, "dataset.json")

    try:
        with open(json_filename, "r") as fp:
            experiment_data = json.load(fp)
    except IOError as e:
        raise Exception("File {} doesn't exist. It should be part of the "
                        "Magna valve directory".format(json_filename))

    return experiment_data


def expand_list(data_path, format=None):
    """Expand the list."""
    sub_folders = os.listdir(data_path)
    data = []
    for sub_folder in sub_folders:
        root = os.path.join(data_path, sub_folder)
        sub_ = os.listdir(root)
        for i, sub in enumerate(sub_):
            if format is not None:
                data.append(glob.glob(os.path.join(root, sub) + format))
            else:
                data.append(os.path.join(root, sub))
    data = natsorted(data)
    return data


def load_scan(path):
    """Load scan."""
    files = os.listdir(path)
    file_dcm = []
    for f in files:
        if f.endswith('.dcm'):
            file_dcm.append(f)
    slices = [pydicom.read_file(path + '/' + s) for s in file_dcm]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] -
            slices[1].ImagePositionPatient[2])
    except Exception as e:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def load_mask(path):
    """Load mask."""
    mask = glob.glob(path + '/*.npy')
    mask = natsorted(mask)

    return mask
