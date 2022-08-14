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
import bz2
import pickle
import os
from scipy.ndimage import measurements, morphology


def agatston_score_slice(
    image, mask_agatston, index, area, threshold_min=None, threshold_max=None
):
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
            if number_of_pix * area <= 1:
                prediction[lw == j] = 0
    prediction = np.ma.masked_where(prediction == 0, prediction)
    return prediction


def area_measurements(slice_):
    """Estimate area."""
    # Generate a structuring element that will consider
    # features connected even if they touch diagonally
    s = morphology.generate_binary_structure(2, 2)
    lw, num = measurements.label(slice_, structure=s)
    area_ = measurements.sum(slice_, lw, index=np.arange(lw.max() + 1))
    return area_, lw


def agatston_score(
    image, mask_agatston, area, threshold_min=None, threshold_max=None
):
    """Get Agatston score."""
    score = 0.0
    for i in range(len(image)):
        prediction = image[i].copy()
        prediction[mask_agatston[i] == 0] = 0
        if threshold_min is not None:
            prediction[prediction < threshold_min] = 0
        if threshold_max is not None:
            prediction[prediction > threshold_max] = 0

        measurement = prediction.copy()
        measurement[measurement > 0] = 1
        area_, lw = area_measurements(measurement)
        for j, number_of_pix in enumerate(area_):
            if j != 0:
                # density higher than 1mm2
                if number_of_pix * area <= 1:
                    prediction[lw == j] = 0

        prediction[prediction > 400] = 4
        prediction[prediction >= 300] = 3
        prediction[prediction >= 200] = 2
        prediction[prediction >= 130] = 1
        score += area * np.sum(prediction)
    return score


def save_prediction(
    image,
    mask_agatston,
    path,
    score,
    area,
    threshold_min,
    threshold_max,
    valve,
    candidate,
    online=False,
):
    """Save prediction in dictionnary."""
    save_predict = {}
    save_predict["score"] = score
    save_predict["image"] = image
    save_predict["area"] = area
    save_predict["mask_agatston"] = mask_agatston
    save_predict["threshold_min"] = threshold_min
    save_predict["threshold_max"] = threshold_max
    save_predict["valve"] = valve
    save_predict["candidate"] = candidate
    if not online:
        save_predict["data_path"] = "/".join(
            [path.split("/")[-2], path.split("/")[-1]]
        )
        folder = path.replace("datasets_dcm", "predictions")
    else:
        folder = path
    os.makedirs(folder, exist_ok=True)
    with bz2.BZ2File(folder + "/prediction.pbz2", "wb") as f:
        pickle.dump(save_predict, f)
    return save_predict
