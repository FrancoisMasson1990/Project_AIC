#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Main script to convert dcm file into a nibabel object.
"""

import aic.misc.files as fs
import aic.misc.utils as ut
import aic.processing.operations as op
import aic.misc.converter as co
import os


if __name__ == "__main__":

    path = str(fs.get_dataset_root())
    folders = os.listdir(path)
    for folder in folders:
        files_ = os.path.join(path, folder)
        file_ = os.listdir(files_)
        for f in file_:
            if os.path.isdir(os.path.join(files_, f)):
                f = os.path.join(files_, f)
                imgs = ut.load_scan(f)
                imgs = op.get_pixels_hu(imgs)
                co.npy_to_nii(imgs,
                              name="imaging.nii.gz",
                              directory=files_)
