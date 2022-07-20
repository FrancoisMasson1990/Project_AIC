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
3D Unet Architecture.

This module contains all of the model definition code.
You can try custom models by modifying the code here.
"""

import os
from tensorflow import keras as K
import aic.processing.metrics as mt


class unet(object):
    """3D U-Net model class."""

    def __init__(self,
                 channels_first=None,
                 filters=None,
                 use_upsampling=None,
                 learning_rate=None,
                 weight_dice_loss=None,
                 output_path=None,
                 inference_filename=None,
                 blocktime=None,
                 num_threads=None,
                 num_inter_threads=None,
                 print_model=None):
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

        self.filters = filters

        self.learningrate = learning_rate
        self.weight_dice_loss = weight_dice_loss

        print("Data format = " + self.data_format)
        K.backend.set_image_data_format(self.data_format)

        self.output_path = output_path
        self.inference_filename = inference_filename

        self.metrics = [self.dice_coef, self.soft_dice_coef]
        # self.loss = self.dice_coef_loss
        self.loss = self.combined_dice_coef_loss

        # Tversky method
        # self.metrics = [self.tversky]
        # self.loss = self.focal_tversky_loss

        self.optimizer = K.optimizers.Adam(learning_rate=self.learningrate)

        self.custom_objects = {
            "combined_dice_coef_loss": self.combined_dice_coef_loss,
            "dice_coef_loss": self.dice_coef_loss,
            "dice_coef": self.dice_coef,
            "soft_dice_coef": self.soft_dice_coef,
            "focal_tversky_loss": self.focal_tversky_loss,
            "tversky": self.tversky}

        self.blocktime = blocktime
        self.num_threads = num_threads
        self.num_inter_threads = num_inter_threads

        self.use_upsampling = use_upsampling
        self.print_model = print_model

    def dice_coef(self,
                  target,
                  prediction,
                  axis=(1, 2, 3)):
        """Get the Sorenson Dice."""
        return mt.dice_coef(target=target,
                            prediction=prediction,
                            axis=axis)

    def soft_dice_coef(self,
                       target,
                       prediction,
                       axis=(1, 2, 3)):
        """Get the Sorenson (Soft) Dice."""
        return mt.soft_dice_coef(target=target,
                                 prediction=prediction,
                                 axis=axis)

    def dice_coef_loss(self,
                       target,
                       prediction,
                       axis=(1, 2, 3)):
        """Get the Sorenson (Soft) Dice loss.

        Using -log(Dice) as the loss since it is better behaved.
        Also, the log allows avoidance of the division which
        can help prevent underflow when the numbers are very small.
        """
        return mt.dice_coef_loss(target=target,
                                 prediction=prediction,
                                 axis=axis)

    def combined_dice_coef_loss(self,
                                target,
                                prediction,
                                axis=(1, 2, 3)):
        """Combine Dice and Binary Cross Entropy Loss."""
        return mt.combined_dice_coef_loss(
            weight_dice_loss=self.weight_dice_loss,
            target=target,
            prediction=prediction,
            axis=axis)

    def tversky(self,
                target,
                prediction):
        """Get tversky loss."""
        return mt.tversky(target=target,
                          prediction=prediction,
                          channels_first=self.channels_first
                          )

    def focal_tversky_loss(self,
                           target,
                           prediction):
        """Get focal tversky loss."""
        return mt.focal_tversky_loss(target=target,
                                     prediction=prediction)

    def convolution_block(self, x, name, filters, params):
        """Get Convolution Block.

        Convolutional block of layers
        Per the original paper this is back to back 3D convs
        with batch norm and then ReLU.
        """
        x = K.layers.Conv3D(filters=filters, **params, name=name+"_conv0")(x)
        x = K.layers.BatchNormalization(name=name+"_bn0")(x)
        x = K.layers.Activation("relu", name=name+"_relu0")(x)

        x = K.layers.Conv3D(filters=filters, **params, name=name+"_conv1")(x)
        x = K.layers.BatchNormalization(name=name+"_bn1")(x)
        x = K.layers.Activation("relu", name=name)(x)
        return x

    def unet_model(self, imgs_shape, msks_shape,
                   final=False):
        """Define the UNet model."""
        num_chan_in = imgs_shape[self.concat_axis]
        num_chan_out = msks_shape[self.concat_axis]
        self.input_shape = imgs_shape
        self.num_input_channels = num_chan_in

        inputs = K.layers.Input(shape=self.input_shape, name="DicomImages")

        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="same",
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(kernel_size=(2, 2, 2),
                            strides=(2, 2, 2),
                            padding="same",
                            kernel_initializer="he_uniform")

        # BEGIN - Encoding path
        encodeA = self.convolution_block(inputs,
                                         "encodeA",
                                         self.filters,
                                         params)
        poolA = K.layers.MaxPooling3D(name="poolA",
                                      pool_size=(2, 2, 2))(encodeA)

        encodeB = self.convolution_block(poolA,
                                         "encodeB",
                                         self.filters*2,
                                         params)
        poolB = K.layers.MaxPooling3D(name="poolB",
                                      pool_size=(2, 2, 2))(encodeB)

        encodeC = self.convolution_block(poolB,
                                         "encodeC",
                                         self.filters*4,
                                         params)
        poolC = K.layers.MaxPooling3D(name="poolC",
                                      pool_size=(2, 2, 2))(encodeC)

        encodeD = self.convolution_block(poolC,
                                         "encodeD",
                                         self.filters*8,
                                         params)
        poolD = K.layers.MaxPooling3D(name="poolD",
                                      pool_size=(2, 2, 2))(encodeD)

        encodeE = self.convolution_block(poolD,
                                         "encodeE",
                                         self.filters*16,
                                         params)
        # END - Encoding path

        # BEGIN - Decoding path
        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2))(encodeE)
        else:
            up = K.layers.Conv3DTranspose(name="transconvE",
                                          filters=self.filters*8,
                                          **params_trans)(encodeE)
        concatD = K.layers.concatenate(
            [up, encodeD], axis=self.concat_axis, name="concatD")

        decodeC = self.convolution_block(concatD,
                                         "decodeC",
                                         self.filters*8,
                                         params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2))(decodeC)
        else:
            up = K.layers.Conv3DTranspose(name="transconvC",
                                          filters=self.filters*4,
                                          **params_trans)(decodeC)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = self.convolution_block(concatC,
                                         "decodeB",
                                         self.filters*4,
                                         params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2))(decodeB)
        else:
            up = K.layers.Conv3DTranspose(name="transconvB",
                                          filters=self.filters*2,
                                          **params_trans)(decodeB)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = self.convolution_block(concatB,
                                         "decodeA",
                                         self.filters*2,
                                         params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2))(decodeA)
        else:
            up = K.layers.Conv3DTranspose(name="transconvA",
                                          filters=self.filters,
                                          **params_trans)(decodeA)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        # END - Decoding path

        convOut = self.convolution_block(concatA,
                                         "convOut",
                                         self.filters,
                                         params)

        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=num_chan_out,
                                     kernel_size=(1, 1, 1),
                                     activation="sigmoid")(convOut)

        model = K.models.Model(inputs=[inputs], outputs=[
                               prediction], name="3DUNet_Valve_Challenge")

        optimizer = self.optimizer

        if final:
            model.trainable = False
        else:
            model.compile(optimizer=optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

            if self.print_model:
                model.summary()

        return model

    def get_callbacks(self):
        """Define any callbacks for the training."""
        model_filename = os.path.join(
            self.output_path, self.inference_filename)

        print("Writing model to '{}'".format(model_filename))

        # Save model whenever we get better validation loss
        model_checkpoint = K.callbacks.ModelCheckpoint(model_filename,
                                                       verbose=1,
                                                       monitor="val_loss",
                                                       save_best_only=True)

        directoryName = \
            "unet_block{}_inter{}_intra{}".format(self.blocktime,
                                                  self.num_threads,
                                                  self.num_inter_threads)

        # Tensorboard callbacks
        if (self.use_upsampling):
            tensorboard_filename = \
                os.path.join(self.output_path,
                             "keras_tensorboard_upsampling/{}".format(
                                 directoryName))
        else:
            tensorboard_filename = \
                os.path.join(self.output_path,
                             "keras_tensorboard/{}".format(
                                 directoryName))

        tensorboard_checkpoint = K.callbacks.TensorBoard(
            log_dir=tensorboard_filename,
            write_graph=True, write_images=True)

        early_stopping = K.callbacks.EarlyStopping(patience=5,
                                                   restore_best_weights=True)

        return model_filename, [model_checkpoint, early_stopping,
                                tensorboard_checkpoint]

    def evaluate_model(self, model_filename, ds_test):
        """Evaluate the best model on the validation dataset."""
        model = K.models.load_model(
            model_filename, custom_objects=self.custom_objects)

        print("Evaluating model on test dataset. Please wait...")
        metrics = model.evaluate(ds_test,
                                 verbose=1)

        for idx, metric in enumerate(metrics):
            print("Test dataset {} = {:.4f}".format(
                model.metrics_names[idx], metric))

    def create_model(self, imgs_shape, msks_shape,
                     final=False):
        """Create model.

        If you have other models, you can try them here
        """
        return self.unet_model(imgs_shape,
                               msks_shape,
                               final=final)

    def load_model(self, model_filename):
        """Load a model from Keras file."""
        return K.models.load_model(model_filename,
                                   custom_objects=self.custom_objects)
