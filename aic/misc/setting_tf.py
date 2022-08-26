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

import os

import psutil
import tensorflow as tf


def requirements_2d():
    """Assign requirements for 2d training."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

    blocktime = 0
    num_inter_threads = 1
    num_threads = min(
        len(psutil.Process().cpu_affinity()), psutil.cpu_count(logical=False)
    )

    os.environ["KMP_BLOCKTIME"] = str(blocktime)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["INTRA_THREADS"] = str(num_threads)
    os.environ["INTER_THREADS"] = str(num_inter_threads)
    os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        print("allowing growth")
        growth = True
    else:
        print("nogrowth")
        growth = False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, growth)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
            )
    except RuntimeError as e:
        print(e)

    return blocktime, num_inter_threads, num_threads


def requirements_3d():
    """Assign requirements for 2d training."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

    blocktime = 0
    num_inter_threads = 1
    num_threads = min(
        len(psutil.Process().cpu_affinity()), psutil.cpu_count(logical=False)
    )

    os.environ["KMP_BLOCKTIME"] = str(blocktime)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["INTRA_THREADS"] = str(num_threads)
    os.environ["INTER_THREADS"] = str(num_inter_threads)
    os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        print("allowing growth")
        growth = True
    else:
        print("nogrowth")
        growth = False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, growth)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
            )
    except RuntimeError as e:
        print(e)

    return blocktime, num_inter_threads, num_threads
