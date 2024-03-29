#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Convert format files
"""

import os

import nibabel as nib
import numpy as np


def npy_to_nii(obj, name="imaging.nii.gz", directory=".", affine=np.eye(4)):
    ni_img = nib.Nifti1Image(obj, affine, dtype=np.uint8)
    save = os.path.join(directory, name)
    nib.save(ni_img, save)
