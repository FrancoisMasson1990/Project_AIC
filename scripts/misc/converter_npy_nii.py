#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Main script to convert npy file into a nibabel object.
"""

import os

import numpy as np

import aic.misc.converter as co
import aic.misc.files as fs
import aic.misc.utils as ut

if __name__ == "__main__":

    path = str(fs.get_label_2d_root())
    save_path = str(fs.get_dataset_root())
    folders = os.listdir(path)
    for folder in folders:
        files_ = os.path.join(path, folder)
        file_ = os.listdir(files_)
        for f in file_:
            if os.path.isdir(os.path.join(files_, f)):
                f = os.path.join(files_, f)
                labels = ut.load_mask(f)
                labels = [np.load(labels[i]) for i in range(len(labels))]
                labels = np.stack(labels, axis=0)
                # labels[labels > 0] = 1.0
                save = os.path.join(save_path, files_.split("/")[-1])
                co.npy_to_nii(
                    labels, name="segmentation.nii.gz", directory=save
                )
