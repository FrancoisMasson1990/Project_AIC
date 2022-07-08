#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Metrics for the model architecture.
"""

import tensorflow as tf
from tensorflow import keras as K


def dice_coef(target,
              prediction,
              axis=(1, 2),
              smooth=0.0001):
    """Get the Sorenson Dice."""
    prediction = K.backend.round(prediction)  # Round to 0 or 1
    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def soft_dice_coef(target,
                   prediction,
                   axis=(1, 2),
                   smooth=0.0001):
    """Get the Sorenson (Soft) Dice."""
    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_coef_loss(target,
                   prediction,
                   axis=(1, 2),
                   smooth=0.0001):
    """Get the Sorenson (Soft) Dice loss.

    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

    return dice_loss


def combined_dice_coef_loss(weight_dice_loss,
                            target,
                            prediction,
                            axis=(1, 2),
                            smooth=0.0001):
    """Combine Dice and Binary Cross Entropy Loss."""
    return weight_dice_loss*dice_coef_loss(target,
                                           prediction,
                                           axis,
                                           smooth) + \
        (1-weight_dice_loss)*K.losses.binary_crossentropy(target,
                                                          prediction)


def tversky(target,
            prediction,
            smooth=1,
            alpha=0.7,
            channels_first=True):
    """Get the Tversky index."""
    # Flatten the input data
    if channels_first:
        y_true = K.backend.permute_dimensions(target,
                                              (3, 1, 2, 0))
        y_pred = K.backend.permute_dimensions(prediction,
                                              (3, 1, 2, 0))
    else:
        y_true = target
        y_pred = prediction

    y_true_pos = K.backend.flatten(y_true)
    y_pred_pos = K.backend.flatten(y_pred)
    true_pos = K.backend.sum(y_true_pos * y_pred_pos)
    false_neg = K.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.backend.sum((1 - y_true_pos) * y_pred_pos)

    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def focal_tversky_loss(target,
                       prediction,
                       gamma=1.25):
    """Get the Tversky focal loss."""
    tv = tversky(target, prediction)

    return K.backend.pow((1 - tv), gamma)
