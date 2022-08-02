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

import numpy as np
import tensorflow as tf
from tensorflow import keras as K


def dice_loss(target,
              prediction,
              axis,
              delta=0.5,
              smooth=0.000001,
              ):
    """Estimate Dice loss originates from Sørensen Dice coefficient.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives.
    smooth : float, optional
        smoothing constant to prevent division by zero errors.
    """
    def loss_function(y_true, y_pred):
        tp = K.backend.sum(y_true * y_pred, axis=axis)
        fn = K.backend.sum(y_true * (1-y_pred), axis=axis)
        fp = K.backend.sum((1-y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        dice_loss = K.backend.mean(1-dice_class)

        return dice_loss

    return loss_function(target, prediction)


def tversky_loss(target,
                 prediction,
                 axis,
                 delta=0.7,
                 smooth=0.000001):
    """Get the Tversky loss function for image segmentation.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives.
    smooth : float, optional
        smoothing constant to prevent division by zero errors.
    """
    def loss_function(y_true, y_pred):
        tp = K.backend.sum(y_true * y_pred, axis=axis)
        fn = K.backend.sum(y_true * (1-y_pred), axis=axis)
        fp = K.backend.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        tversky_loss = K.backend.mean(1-tversky_class)

        return tversky_loss

    return loss_function(target, prediction)


def dice_coefficient(target,
                     prediction,
                     axis,
                     delta=0.5,
                     smooth=0.000001):
    """Get the Dice similarity coefficient.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives.
    smooth : float, optional
        smoothing constant to prevent division by zero errors.
    """
    tp = K.backend.sum(target * prediction, axis=axis)
    fn = K.backend.sum(target * (1-prediction), axis=axis)
    fp = K.backend.sum((1-target) * prediction, axis=axis)
    dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Average class scores
    dice = K.backend.mean(dice_class)

    return dice


def combo_loss(target,
               prediction,
               axis,
               alpha=0.5,
               beta=0.5):
    """Get the Combo Loss.

    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss.
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives.
    """
    def loss_function(y_true, y_pred):
        dice = dice_coefficient(y_true,
                                y_pred,
                                axis=axis)
        epsilon = K.backend.epsilon()
        y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.backend.log(y_pred)
        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = K.backend.mean(K.backend.sum(cross_entropy, axis=[-1]))
        if alpha is not None:
            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss

    return loss_function(target, prediction)


def focal_tversky_loss(target,
                       prediction,
                       axis,
                       delta=0.7,
                       gamma=0.75,
                       smooth=0.000001):
    """Get the novel Focal Tversky loss function with improved Attention U-Net.

    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples.
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.backend.epsilon()
        y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
        tp = K.backend.sum(y_true * y_pred, axis=axis)
        fn = K.backend.sum(y_true * (1-y_pred), axis=axis)
        fp = K.backend.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        focal_tversky_loss = K.backend.mean(
            K.backend.pow((1-tversky_class), gamma))

        return focal_tversky_loss

    return loss_function(target, prediction)


def focal_loss(target,
               prediction,
               alpha=None,
               gamma_f=2.):
    """Get the Focal loss to address the issue of the class imbalance problem.

    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives.
        alpha > 0.5 penalises false negatives more than false positives,
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples.
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.backend.epsilon()
        y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.backend.log(y_pred)
        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * \
                K.backend.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.backend.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.backend.mean(K.backend.sum(focal_loss, axis=[-1]))

        return focal_loss

    return loss_function(target, prediction)


def symmetric_focal_loss(target,
                         prediction,
                         delta=0.7,
                         gamma=2.):
    """
    Get the Symmetric Focal loss.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives.
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting.
    """
    def loss_function(y_true, y_pred):
        epsilon = K.backend.epsilon()
        y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.backend.log(y_pred)
        # calculate losses separately for each class
        back_ce = K.backend.pow(1 - y_pred[:, :, :, 0],
                                gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - delta) * back_ce

        fore_ce = K.backend.pow(1 - y_pred[:, :, :, 1],
                                gamma) * cross_entropy[:, :, :, 1]
        fore_ce = delta * fore_ce

        loss = K.backend.mean(
            K.backend.sum(tf.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss

    return loss_function(target, prediction)


def symmetric_focal_tversky_loss(target,
                                 prediction,
                                 axis,
                                 delta=0.7,
                                 gamma=0.75):
    """Get the symmetric focal tversky loss.

    This is the implementation for binary segmentation.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives.
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples.
    """
    def loss_function(y_true, y_pred):
        epsilon = K.backend.epsilon()
        y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
        tp = K.backend.sum(y_true * y_pred, axis=axis)
        fn = K.backend.sum(y_true * (1-y_pred), axis=axis)
        fp = K.backend.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        # calculate losses separately for each class, enhancing both classes
        back_dice = \
            (1-dice_class[:, 0]) * K.backend.pow(1-dice_class[:, 0], -gamma)
        fore_dice = \
            (1-dice_class[:, 1]) * K.backend.pow(1-dice_class[:, 1], -gamma)

        # Average class scores
        loss = K.backend.mean(tf.stack([back_dice, fore_dice], axis=-1))
        return loss

    return loss_function(target, prediction)


def asymmetric_focal_loss(target,
                          prediction,
                          delta=0.7,
                          gamma=2.):
    """Get the asymmetric focal loss for Imbalanced datasets.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives.
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting
        of easy examples.
    """
    def loss_function(y_true, y_pred):
        epsilon = K.backend.epsilon()
        y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.backend.log(y_pred)
        back_ce = K.backend.pow(
            1 - y_pred[:, :, :, 0], gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - delta) * back_ce
        fore_ce = cross_entropy[:, :, :, 1]
        fore_ce = delta * fore_ce

        loss = K.backend.mean(
            K.backend.sum(tf.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss

    return loss_function(target, prediction)


def asymmetric_focal_tversky_loss(target,
                                  prediction,
                                  axis,
                                  delta=0.7,
                                  gamma=0.75):
    """Get the asymmetric focal loss for binary segmentation.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives.
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples.
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.backend.epsilon()
        y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
        tp = K.backend.sum(y_true * y_pred, axis=axis)
        fn = K.backend.sum(y_true * (1-y_pred), axis=axis)
        fp = K.backend.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
        back_dice = (1-dice_class[:, 0])
        fore_dice = \
            (1-dice_class[:, 1]) * K.backend.pow(1-dice_class[:, 1], -gamma)
        # Average class scores
        loss = K.backend.mean(tf.stack([back_dice, fore_dice], axis=-1))
        return loss

    return loss_function(target, prediction)


def sym_unified_focal_loss(target,
                           prediction,
                           weight=0.5,
                           delta=0.6,
                           gamma=0.5):
    """Get the Unified Focal loss.

    This a new compound loss function that unifies
    Dice-based and cross entropy-based loss functions into a single framework.

    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight
        given to symmetric Focal Tversky loss
        and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background
        suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true, y_pred):
        symmetric_ftl = symmetric_focal_tversky_loss(
            y_true, y_pred, delta=delta, gamma=gamma)
        symmetric_fl = symmetric_focal_loss(
            y_true, y_pred, delta=delta, gamma=gamma)
        if weight is not None:
            return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)
        else:
            return symmetric_ftl + symmetric_fl

    return loss_function(target, prediction)


def asym_unified_focal_loss(target,
                            prediction,
                            weight=0.5,
                            delta=0.6,
                            gamma=0.5):
    """Get the Unified Focal loss.

    This is a new compound loss function that unifies Dice-based
    and cross entropy-based loss functions into a single framework.

    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight
        given to asymmetric Focal Tversky loss
        and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class.
    gamma : float, optional
        focal parameter controls the degree
        of background suppression and foreground enhancement.
    """
    def loss_function(y_true, y_pred):
        asymmetric_ftl = asymmetric_focal_tversky_loss(
            y_true, y_pred, delta=delta, gamma=gamma)
        asymmetric_fl = asymmetric_focal_loss(
            y_true, y_pred, delta=delta, gamma=gamma)
        if weight is not None:
            return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

    return loss_function(target, prediction)
