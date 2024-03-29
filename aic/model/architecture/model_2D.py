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
2D Unet Architecture.

This module contains all of the model definition code.
You can try custom models by modifying the code here.
"""

import os

from tensorflow import keras as K

import aic.processing.metrics as mt


class Unet(object):
    """2D U-Net model class."""

    def __init__(
        self,
        channels_first=None,
        fms=None,
        output_path=None,
        inference_filename=None,
        blocktime=None,
        num_threads=None,
        learning_rate=None,
        weight_dice_loss=None,
        num_inter_threads=None,
        use_upsampling=None,
        use_dropout=None,
        print_model=None,
    ):
        """Init class."""
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
        self.weight_dice_loss = weight_dice_loss

        print("Data format = " + self.data_format)
        K.backend.set_image_data_format(self.data_format)

        self.output_path = output_path
        self.inference_filename = inference_filename
        self.axis = (1, 2)

        self.metrics = [self.dice_coef]
        # self.loss = self.dice_coef_loss
        # self.loss = self.combined_dice_coef_loss

        # Tversky method
        self.loss = self.focal_tversky_loss

        self.optimizer = K.optimizers.Adam(learning_rate=self.learningrate)

        self.custom_objects = {
            "combined_dice_ce_loss": self.combined_dice_coef_loss,
            "dice_coef_loss": self.dice_coef_loss,
            "dice_coef": self.dice_coef,
            "focal_tversky_loss": self.focal_tversky_loss,
        }

        self.blocktime = blocktime
        self.num_threads = num_threads
        self.num_inter_threads = num_inter_threads

        self.use_upsampling = use_upsampling
        self.use_dropout = use_dropout
        self.print_model = print_model

    def dice_coef(self, target, prediction):
        """Get the Sorenson Dice."""
        return mt.dice_coefficient(
            target=target, prediction=prediction, axis=self.axis
        )

    def dice_coef_loss(self, target, prediction):
        """Get the Sorenson (Soft) Dice loss.

        Using -log(Dice) as the loss since it is better behaved.
        Also, the log allows avoidance of the division which
        can help prevent underflow when the numbers are very small.
        """
        return mt.dice_loss(
            target=target, prediction=prediction, axis=self.axis
        )

    def combined_dice_coef_loss(self, target, prediction):
        """Combine Dice and Binary Cross Entropy Loss."""
        return mt.combo_loss(
            target=target,
            prediction=prediction,
            axis=self.axis,
        )

    def focal_tversky_loss(self, target, prediction):
        """Get focal tversky loss."""
        return mt.focal_tversky_loss(
            target=target, prediction=prediction, axis=self.axis
        )

    def unet_model(self, imgs_shape, msks_shape, dropout=0.2, final=False):
        """Define the UNet model.

        U-Net Model
        ===========
        Based on https://arxiv.org/abs/1505.04597
        The default uses UpSampling2D (nearest neighbors interpolation) in
        the decoder path. The alternative is to use Transposed
        Convolution.
        """
        if not final:
            if self.use_upsampling:
                print("Using UpSampling2D")
            else:
                print("Using Transposed Convolution")

        num_chan_in = imgs_shape[self.concat_axis]
        num_chan_out = msks_shape[self.concat_axis]

        # You can make the network work on variable input height and width
        # if you pass None as the height and width

        self.input_shape = imgs_shape

        self.num_input_channels = num_chan_in

        inputs = K.layers.Input(self.input_shape, name="DicomImages")

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
            up = K.layers.UpSampling2D(name="upE", size=(2, 2))(encodeE)
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
            up = K.layers.UpSampling2D(name="upC", size=(2, 2))(decodeC)
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
            up = K.layers.UpSampling2D(name="upB", size=(2, 2))(decodeB)
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
            up = K.layers.UpSampling2D(name="upA", size=(2, 2))(decodeA)
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

        model = K.models.Model(
            inputs=[inputs],
            outputs=[prediction],
            name="2DUNet_Valve_Challenge",
        )

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

    def get_callbacks(self):
        """Define any callbacks for the training."""
        model_filename = os.path.join(
            self.output_path, self.inference_filename
        )

        print("Writing model to '{}'".format(model_filename))

        # Save model whenever we get better validation loss
        model_checkpoint = K.callbacks.ModelCheckpoint(
            model_filename, verbose=1, monitor="val_loss", save_best_only=True
        )

        directoryName = "unet_block{}_inter{}_intra{}".format(
            self.blocktime, self.num_threads, self.num_inter_threads
        )

        # Tensorboard callbacks
        if self.use_upsampling:
            tensorboard_filename = os.path.join(
                self.output_path,
                "keras_tensorboard_upsampling/{}".format(directoryName),
            )
        else:
            tensorboard_filename = os.path.join(
                self.output_path,
                "keras_tensorboard_transposed/{}".format(directoryName),
            )

        tensorboard_checkpoint = K.callbacks.TensorBoard(
            log_dir=tensorboard_filename, write_graph=True, write_images=True
        )

        early_stopping = K.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        )

        return model_filename, [
            model_checkpoint,
            early_stopping,
            tensorboard_checkpoint,
        ]

    def evaluate_model(self, model_filename, ds_test):
        """Evaluate the best model on the validation dataset."""
        model = K.models.load_model(
            model_filename, custom_objects=self.custom_objects
        )

        print("Evaluating model on test dataset. Please wait...")
        metrics = model.evaluate(ds_test, verbose=1)

        for idx, metric in enumerate(metrics):
            print(
                "Test dataset {} = {:.4f}".format(
                    model.metrics_names[idx], metric
                )
            )

    def create_model(self, imgs_shape, msks_shape, dropout=0.2, final=False):
        """Create model.

        If you have other models, you can try them here
        """
        return self.unet_model(
            imgs_shape, msks_shape, dropout=dropout, final=final
        )

    def load_model(self, model_filename):
        """Load a model from Keras file."""
        return K.models.load_model(
            model_filename, custom_objects=self.custom_objects
        )
