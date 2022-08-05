#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Convert a Keras model into a TensorFlow Lite model.
"""

import tensorflow as tf


def save_to_tflite(model, name="model.tflite"):
    """Convert and save a Keras model into a TensorFlow Lite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # Save the model.
    with open(name, "wb") as f:
        f.write(tflite_model)
