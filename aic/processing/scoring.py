#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for scoring function
"""

import numpy as np
from scipy.ndimage import measurements


def agatston_score_slice(image,
                         mask_agatston,
                         index,
                         area,
                         threshold_min=130,
                         threshold_max=450):
    """Get Agatston score by slide."""
    prediction = image[index].copy()
    prediction[mask_agatston[index] == 0] = 0
    if threshold_min is not None:
        prediction[prediction < threshold_min] = 0
    if threshold_max is not None:
        prediction[prediction > threshold_max] = 0
    prediction[prediction > 0] = 1
    area_, lw = area_measurements(prediction)
    for j, number_of_pix in enumerate(area_):
        if j != 0:
            # density higher than 1mm2
            if number_of_pix*area <= 1:
                prediction[lw == j] = 0
    prediction = \
        np.ma.masked_where(prediction == 0,
                           prediction)
    return prediction


def area_measurements(slice_):
    """Estimate area."""
    slice_[slice_ != 0] = 1
    lw, num = measurements.label(slice_)
    area_ = measurements.sum(slice_, lw, index=np.arange(lw.max() + 1))
    return area_, lw


def agatston_score(image,
                   mask_agatston,
                   area,
                   threshold_min=130,
                   threshold_max=450):
    """Get Agatston score."""
    score = 0.0
    for i in range(len(image)):
        prediction = image[i].copy()
        prediction[mask_agatston[i] == 0] = 0
        if threshold_min is not None:
            prediction[prediction < threshold_min] = 0
        if threshold_max is not None:
            prediction[prediction > threshold_max] = 0
        area_, lw = area_measurements(prediction)
        for j, number_of_pix in enumerate(area_):
            if j != 0:
                # density higher than 1mm2
                if number_of_pix * area <= 1:
                    prediction[lw == j] = 0

        prediction[np.logical_and(prediction >= 130, prediction < 200)] = 1
        prediction[np.logical_and(prediction >= 200, prediction < 300)] = 2
        prediction[np.logical_and(prediction >= 300, prediction < 400)] = 3
        prediction[prediction > 400] = 4

        score += area*np.sum(prediction)
    return score
