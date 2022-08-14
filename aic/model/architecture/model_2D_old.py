#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

"""
This module contains all of the model definition code.
You can try custom models by modifying the code here.
"""

import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import tensorflow_addons as tfa
import segmentation_models as sm

import tensorflow.keras.backend as backend

# LABEL_CHANNELS = {"labels":{
# 	 			  "background":0,
# 				  "other":1,
# 	 			  "Magna_valve":2,
# 				 }}

LABEL_CHANNELS = {
    "labels": {
        "background": 0,
        "Magna_valve": 1,
    }
}


class unet(object):
    """
    2D U-Net model class
    """

    def __init__(
        self,
        channels_first=None,
        fms=None,
        output_path=None,
        inference_filename=None,
        batch_size=None,
        blocktime=None,
        num_threads=None,
        learning_rate=None,
        weight_dice_loss=None,
        num_inter_threads=None,
        use_upsampling=None,
        use_dropout=None,
        print_model=None,
    ):

        self.channels_first = channels_first
        if self.channels_first:
            """
            Use NCHW format for data
            """
            self.concat_axis = 1
            self.data_format = "channels_first"

        else:
            """
            Use NHWC format for data
            """
            self.concat_axis = -1
            self.data_format = "channels_last"

        self.fms = fms  # 32 or 16 depending on your memory size
        self.learningrate = learning_rate

        print("Data format = " + self.data_format)
        K.backend.set_image_data_format(self.data_format)

        self.output_path = output_path
        self.inference_filename = inference_filename

        self.batch_size = batch_size

        self.metrics = ["accuracy", self.dice_coef, self.soft_dice_coef]
        self.weight_dice_loss = weight_dice_loss

        if self.weight_dice_loss is not None:
            self.loss = self.combined_dice_ce_loss
        else:
            self.loss = self.dice_coef_loss

        self.optimizer = K.optimizers.Adam(learning_rate=self.learningrate)

        self.custom_objects = {
            "combined_dice_ce_loss": self.combined_dice_ce_loss,
            "dice_coef_loss": self.dice_coef_loss,
            "dice_coef": self.dice_coef,
            "soft_dice_coef": self.soft_dice_coef,
        }

        self.blocktime = blocktime
        self.num_threads = num_threads
        self.num_inter_threads = num_inter_threads

        self.use_upsampling = use_upsampling
        self.use_dropout = use_dropout
        self.print_model = print_model

    def dice_coef(self, target, prediction, axis=(1, 2), smooth=0.01):
        """
        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """
        prediction = K.backend.round(prediction)  # Round to 0 or 1

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.0) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)

    def dice_coef_multilabel(self, target, prediction, numLabels=2):
        # This simply calculates the dice score for each individual label,
        # and then sums them together, and includes the background.
        # The best dice score you will ever get is equal to numLables*-1.0.
        # When monitoring I always keep in mind that the dice for the background is almost always near 1.0.
        dice = 0
        for index in range(numLabels):
            dice -= self.dice_coef(
                target[:, :, :, index], prediction[:, :, :, index]
            )
        return dice

    def soft_dice_coef(self, target, prediction, axis=(1, 2), smooth=0.01):
        """
        Sorenson (Soft) Dice  - Don't round the predictions
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.0) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)

    def dice_coef_loss(self, target, prediction, axis=(1, 2), smooth=1.0):
        """
        Sorenson (Soft) Dice loss
        Using -log(Dice) as the loss since it is better behaved.
        Also, the log allows avoidance of the division which
        can help prevent underflow when the numbers are very small.
        """

        intersection = tf.reduce_sum(prediction * target, axis=axis)
        p = tf.reduce_sum(prediction, axis=axis)
        t = tf.reduce_sum(target, axis=axis)
        numerator = tf.reduce_mean(intersection + smooth)
        denominator = tf.reduce_mean(t + p + smooth)
        # dice_loss = -tf.log(2.*numerator) + tf.log(denominator)
        # Log implementation
        dice_loss = -tf.math.log(2.0 * numerator) + tf.math.log(denominator)

        return dice_loss

    def combined_dice_ce_loss(
        self, target, prediction, axis=(1, 2), smooth=1.0
    ):
        """
        Combined Dice and Binary Cross Entropy Loss
        """
        return self.weight_dice_loss * self.dice_coef_loss(
            target, prediction, axis, smooth
        ) + (1 - self.weight_dice_loss) * K.losses.binary_crossentropy(
            target, prediction
        )

    def tversky(self, target, prediction, smooth=1, alpha=0.7):
        # Flatten the input data
        y_true = backend.permute_dimensions(target, (3, 1, 2, 0))
        y_pred = backend.permute_dimensions(prediction, (3, 1, 2, 0))

        y_true_pos = backend.batch_flatten(y_true)
        y_pred_pos = backend.batch_flatten(y_pred)
        true_pos = backend.sum(y_true_pos * y_pred_pos, 1)
        false_neg = backend.sum(y_true_pos * (1 - y_pred_pos), 1)
        false_pos = backend.sum((1 - y_true_pos) * y_pred_pos, 1)
        return (true_pos + smooth) / (
            true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
        )

    def tversky_loss(self, target, prediction):
        return 1 - self.tversky(target, prediction)

    def focal_tversky_loss(self, target, prediction, gamma=1.5):
        tv = self.tversky(target, prediction)
        return backend.pow((1 - tv), gamma)

    def total_coef_loss(self, n_classes, weights=None):
        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        # set class weights for dice_loss (background: 0.5; bones: 1.; valve: 2.0;)
        if weights is None:
            weights = np.array([0.1, 100])

        dice_loss = sm.losses.DiceLoss(class_weights=weights)
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        return total_loss

    def unet_model(self, imgs_shape, msks_shape, dropout=0.2, final=False):
        """
        U-Net Model
        ===========
        Based on https://arxiv.org/abs/1505.04597
        The default uses UpSampling2D (bilinear interpolation) in
        the decoder path. The alternative is to use Transposed
        Convolution.
        """

        if not final:
            if self.use_upsampling:
                print("Using UpSampling2D")
            else:
                print("Using Transposed Deconvolution")

        num_chan_in = imgs_shape[self.concat_axis]
        num_chan_out = msks_shape[self.concat_axis]

        # You can make the network work on variable input height and width
        # if you pass None as the height and width
        if self.channels_first:
            self.input_shape = [num_chan_in, None, None]
        else:
            self.input_shape = [None, None, num_chan_in]

        # self.input_shape = imgs_shape[1:]

        self.num_input_channels = num_chan_in

        inputs = K.layers.Input(self.input_shape, name="MRImages")

        # Convolution parameters
        params = dict(
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_uniform",
        )

        # Transposed convolution parameters
        params_trans = dict(kernel_size=(2, 2), strides=(2, 2), padding="same")

        encodeA = K.layers.Conv2D(name="encodeAa", filters=self.fms, **params)(
            inputs
        )
        encodeA = K.layers.Conv2D(name="encodeAb", filters=self.fms, **params)(
            encodeA
        )
        poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

        encodeB = K.layers.Conv2D(
            name="encodeBa", filters=self.fms * 2, **params
        )(poolA)
        encodeB = K.layers.Conv2D(
            name="encodeBb", filters=self.fms * 2, **params
        )(encodeB)
        poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

        encodeC = K.layers.Conv2D(
            name="encodeCa", filters=self.fms * 4, **params
        )(poolB)
        if self.use_dropout:
            encodeC = K.layers.SpatialDropout2D(dropout)(encodeC)
        encodeC = K.layers.Conv2D(
            name="encodeCb", filters=self.fms * 4, **params
        )(encodeC)

        poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)
        # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

        encodeD = K.layers.Conv2D(
            name="encodeDa", filters=self.fms * 8, **params
        )(poolC)
        if self.use_dropout:
            encodeD = K.layers.SpatialDropout2D(dropout)(encodeD)
        encodeD = K.layers.Conv2D(
            name="encodeDb", filters=self.fms * 8, **params
        )(encodeD)

        poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

        encodeE = K.layers.Conv2D(
            name="encodeEa", filters=self.fms * 16, **params
        )(poolD)
        encodeE = K.layers.Conv2D(
            name="encodeEb", filters=self.fms * 16, **params
        )(encodeE)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(
                name="upE", size=(2, 2), interpolation="bilinear"
            )(encodeE)
        else:
            up = K.layers.Conv2DTranspose(
                name="transconvE", filters=self.fms * 8, **params_trans
            )(encodeE)
        concatD = K.layers.concatenate(
            [up, encodeD], axis=self.concat_axis, name="concatD"
        )

        decodeC = K.layers.Conv2D(
            name="decodeCa", filters=self.fms * 8, **params
        )(concatD)
        decodeC = K.layers.Conv2D(
            name="decodeCb", filters=self.fms * 8, **params
        )(decodeC)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(
                name="upC", size=(2, 2), interpolation="bilinear"
            )(decodeC)
        else:
            up = K.layers.Conv2DTranspose(
                name="transconvC", filters=self.fms * 4, **params_trans
            )(decodeC)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC"
        )

        decodeB = K.layers.Conv2D(
            name="decodeBa", filters=self.fms * 4, **params
        )(concatC)
        decodeB = K.layers.Conv2D(
            name="decodeBb", filters=self.fms * 4, **params
        )(decodeB)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(
                name="upB", size=(2, 2), interpolation="bilinear"
            )(decodeB)
        else:
            up = K.layers.Conv2DTranspose(
                name="transconvB", filters=self.fms * 2, **params_trans
            )(decodeB)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB"
        )

        decodeA = K.layers.Conv2D(
            name="decodeAa", filters=self.fms * 2, **params
        )(concatB)
        decodeA = K.layers.Conv2D(
            name="decodeAb", filters=self.fms * 2, **params
        )(decodeA)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(
                name="upA", size=(2, 2), interpolation="bilinear"
            )(decodeA)
        else:
            up = K.layers.Conv2DTranspose(
                name="transconvA", filters=self.fms, **params_trans
            )(decodeA)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA"
        )

        convOut = K.layers.Conv2D(name="convOuta", filters=self.fms, **params)(
            concatA
        )
        convOut = K.layers.Conv2D(name="convOutb", filters=self.fms, **params)(
            convOut
        )

        prediction = K.layers.Conv2D(
            name="PredictionMask",
            filters=num_chan_out,
            kernel_size=(1, 1),
            activation="sigmoid",
        )(convOut)

        # Can not use softmax in multilabel classification
        # https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede
        # prediction = K.layers.Conv2D(name="PredictionMask",
        #                             filters=num_chan_out, kernel_size=(1, 1),
        #                             activation="softmax")(convOut)

        model = K.models.Model(inputs=[inputs], outputs=[prediction])

        optimizer = self.optimizer

        if final:
            model.trainable = False
        else:

            model.compile(
                optimizer=optimizer, loss=self.loss, metrics=self.metrics
            )

            if self.print_model:
                model.summary()

        return model

    def unet_model_multi_label(self, print_model=True):

        sm.set_framework("tf.keras")
        BACKBONE = "efficientnetb3"
        classes = LABEL_CHANNELS["labels"].values()

        preprocess_input = sm.get_preprocessing(BACKBONE)
        # define network parameters
        n_classes = len(classes)
        activation = "sigmoid" if n_classes == 1 else "softmax"

        # create model
        model = sm.Unet(
            BACKBONE,
            classes=n_classes,
            activation=activation,
            encoder_weights=None,
            input_shape=(None, None, 1),
        )
        # define optomizer
        optim = K.optimizers.Adam(self.learningrate)

        loss = self.focal_tversky_loss
        metrics = [self.dice_coef_multilabel]
        model.compile(optim, loss, metrics)

        if print_model:
            model.summary()

        return model

    def get_callbacks(self):
        """
        Define any callbacks for the training
        """

        model_filename = os.path.join(
            self.output_path, self.inference_filename
        )

        print("Writing model to '{}'".format(model_filename))

        # Save model whenever we get better validation loss
        model_checkpoint = K.callbacks.ModelCheckpoint(
            model_filename,
            verbose=1,
            monitor="val_loss",  # mode='min'
            save_best_only=True,
        )

        lr_checkpoint = K.callbacks.ReduceLROnPlateau()

        directoryName = "unet_block{}_inter{}_intra{}".format(
            self.blocktime, self.num_threads, self.num_inter_threads
        )

        # Tensorboard callbacks
        if self.use_upsampling:
            tensorboard_filename = os.path.join(
                self.output_path,
                "keras_tensorboard_upsampling"
                "_batch{}/{}".format(self.batch_size, directoryName),
            )
        else:
            tensorboard_filename = os.path.join(
                self.output_path,
                "keras_tensorboard_transposed"
                "_batch{}/{}".format(self.batch_size, directoryName),
            )

        tensorboard_checkpoint = K.callbacks.TensorBoard(
            log_dir=tensorboard_filename, write_graph=True, write_images=True
        )

        return model_filename, [
            model_checkpoint,
            tensorboard_checkpoint,
            lr_checkpoint,
        ]

    def evaluate_model(self, model_filename, imgs_validation, msks_validation):
        """
        Evaluate the best model on the validation dataset
        """

        try:
            model = K.models.load_model(
                model_filename, custom_objects=self.custom_objects
            )
        except:
            model = self.unet_model_multi_label(print_model=False)
            model.load_weights(model_filename)

        K.backend.set_learning_phase(0)
        start_inference = time.time()
        print("Evaluating model on test dataset. Please wait...")
        metrics = model.evaluate(
            imgs_validation,
            msks_validation,
            batch_size=self.batch_size,
            verbose=1,
        )
        elapsed_time = time.time() - start_inference
        print(
            "{} images in {:.2f} seconds => {:.3f} images per "
            "second inference".format(
                imgs_validation.shape[0],
                elapsed_time,
                imgs_validation.shape[0] / elapsed_time,
            )
        )

        for idx, metric in enumerate(metrics):
            print(
                "Test dataset {} = {:.4f}".format(
                    model.metrics_names[idx], metric
                )
            )

    def create_model(
        self, imgs_shape, msks_shape, intel_model, dropout=0.2, final=False
    ):
        """
        If you have other models, you can try them here
        """
        if intel_model:
            return self.unet_model(
                imgs_shape, msks_shape, dropout=dropout, final=final
            )
        else:
            return self.unet_model_multi_label()

    def load_model(self, model_filename, intel_model=True):
        """
        Load a model from Keras file
        """
        if intel_model:
            return K.models.load_model(
                model_filename, custom_objects=self.custom_objects
            )
        else:
            model = self.unet_model_multi_label(print_model=False)
            model.load_weights(model_filename)
            return model

    def save_frozen_model(self, model_filename, input_shape, intel_model):
        """
        Save frozen TensorFlow formatted model protobuf
        """
        model = self.load_model(model_filename, intel_model)

        # Change filename to protobuf extension
        base = os.path.basename(model_filename)
        output_model = os.path.splitext(base)[0] + ".pb"

        # Set Keras to inference
        K.backend._LEARNING_PHASE = tf.constant(0)
        K.backend.set_learning_phase(False)
        K.backend.set_learning_phase(0)
        K.backend.set_image_data_format("channels_last")

        num_output = len(model.outputs)
        predictions = [None] * num_output
        prediction_node_names = [None] * num_output

        for i in range(num_output):
            prediction_node_names[i] = "output_node" + str(i)
            predictions[i] = tf.identity(
                model.outputs[i], name=prediction_node_names[i]
            )

        sess = K.backend.get_session()

        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), prediction_node_names
        )
        infer_graph = graph_util.remove_training_nodes(constant_graph)

        # Write protobuf of frozen model
        frozen_dir = "./frozen_model/"
        # Remove existing directory
        shutil.rmtree(frozen_dir, ignore_errors=True)
        graph_io.write_graph(
            infer_graph, frozen_dir, output_model, as_text=False
        )

        pb_filename = os.path.join(frozen_dir, output_model)
        print("Frozen TensorFlow model written to: {}".format(pb_filename))
        print("Convert this to OpenVINO by running:\n")
        print("source /opt/intel/openvino/bin/setupvars.sh")
        print(
            "python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\"
        )
        print("       --input_model {} \\".format(pb_filename))

        shape_string = "[1"
        for idx in range(len(input_shape[1:])):
            shape_string += ",{}".format(input_shape[idx + 1])
        shape_string += "]"

        print("       --input_shape {} \\".format(shape_string))
        print("       --output_dir openvino_models/FP32/ \\")
        print("       --data_type FP32\n\n")
