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
import shutil


def get_file_list(data_path, json_filename, seed=816, split=0.7):
    """Get list of files.

    Shuffling of the dataset required for the model training/evaluation
    """
    experiment_data = json_export(json_filename)

    # Print information about the Magna valve experiment data
    print("*" * 30)
    print("=" * 30)
    print("Dataset name:        ", experiment_data["name"])
    print("Dataset description: ", experiment_data["description"])
    print("Tensor image size:   ", experiment_data["tensorImageSize"])
    print("=" * 30)
    print("*" * 30)

    """
    Randomize the file list. Then separate into training and
    validation lists. We won't use the testing set since we
    don't have ground truth masks for this; instead we'll
    split the validation set into separate test and validation
    sets.
    """
    dataset_folder = data_path + experiment_data["dataset_folder"]
    label_folder = data_path + experiment_data["label_folder"]
    image_files = expand_list(dataset_folder)
    label_files = expand_list(label_folder)

    assert len(image_files) == len(
        label_files
    ), "Files and labels don't have the same length"

    # Set the random seed so that always get same random mix
    np.random.seed(seed)
    numFiles = len(image_files)
    idxList = np.arange(numFiles)  # List of file indices
    # Shuffle the indices to randomize train/test/split
    np.random.shuffle(idxList)

    # index for the end of the training files
    trainIdx = int(np.floor(numFiles * split))
    trainList = idxList[:trainIdx]

    otherList = idxList[trainIdx:]
    numOther = len(otherList)
    otherIdx = numOther // 2  # index for the end of the testing files
    validateList = otherList[:otherIdx]
    testList = otherList[otherIdx:]

    if len(validateList) == 0:
        validateList = testList
    if len(trainList) == 0:
        trainList = testList
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

    return (
        trainFiles,
        trainLabels,
        validateFiles,
        validateLabels,
        testFiles,
        testLabels,
    )


def slice_file_list(data_path, json_filename):
    """Get the max number of slice in the dataset.

    Required for the Dataloader Generator
    """
    experiment_data = json_export(json_filename)
    dataset_folder = data_path + experiment_data["dataset_folder"]
    num_slice_max = expand_list(dataset_folder, format="/*.dcm")
    num_slice_max = max(len(x) for x in num_slice_max)

    return num_slice_max


def json_export(json_filename):
    """Extract dataset informations contained into a json file."""
    try:
        with open(json_filename, "r") as fp:
            experiment_data = json.load(fp)
    except IOError as e:
        raise Exception(
            "File {} doesn't exist. It should be part of the "
            "Magna valve directory".format(json_filename)
        )

    return experiment_data


def expand_list(data_path, format=None):
    """Expand the list."""
    sub_folders = os.listdir(data_path)
    data = []
    for sub_folder in sub_folders:
        root = os.path.join(data_path, sub_folder)
        sub_ = os.listdir(root)
        for i, sub in enumerate(sub_):
            if os.path.isfile(os.path.join(root, sub)):
                continue
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
        if f.endswith(".dcm"):
            file_dcm.append(f)
    return get_slices(file_dcm, path)


def get_slices(file_dcm, path=None):
    """Get Slice metadata."""
    if path:
        slices = [pydicom.read_file(path + "/" + s) for s in file_dcm]
    else:
        slices = [pydicom.read_file(s) for s in file_dcm]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2]
            - slices[1].ImagePositionPatient[2]
        )
    except Exception as e:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation
        )

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def load_mask(path):
    """Load mask."""
    mask = glob.glob(path + "/*.npy")
    mask = natsorted(mask)

    return mask


def save_dicom(files_dcm, path="./cache/tmp"):
    """Save to pydicom object to dcm format."""
    if not isinstance(files_dcm, list):
        files_dcm = [files_dcm]
    slices = [pydicom.read_file(s) for s in files_dcm]
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    for i, s in enumerate(slices):
        s.save_as(f"{path}/{i}.dcm")
