#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library using dataloader mindset to load
big dataset.
"""

import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
from aic.processing import preprocess as dp
import aic.misc.utils as ut
import aic.processing.operations as op


class DatasetGenerator2D(Sequence):
    """TensorFlow Dataset from Python/NumPy Iterator."""

    def __init__(self,
                 filenames,
                 labelnames,
                 num_slices_per_scan,
                 batch_size=8,
                 crop_dim=[240, 240],
                 augment=False,
                 seed=816,
                 imbalanced=False,
                 z_slice_min=-1,
                 z_slice_max=-1):
        """Init function."""
        img = ut.load_scan(filenames[0])
        img = op.get_pixels_hu(img)
        # We'll assume z-dimension (slice)
        # is last so we will invert dim in the batch
        self.slice_dim = 2
        # Determine the number of slices
        # (we'll assume this is consistent for the other images)
        self.num_slices_per_scan = num_slices_per_scan
        self.imbalanced = imbalanced

        # If crop_dim == -1, then don't crop
        if crop_dim[0] == -1:
            crop_dim[0] = img.shape[1]
        if crop_dim[1] == -1:
            crop_dim[1] = img.shape[2]
        self.crop_dim = crop_dim

        # If crop_dim in z
        self.z_slice_min = z_slice_min
        self.z_slice_max = z_slice_max

        self.filenames = filenames
        self.labelnames = labelnames
        self.batch_size = batch_size

        self.augment = augment
        self.seed = seed

        self.num_files = len(self.filenames)

        self.ds = self.get_dataset()

    def augment_data(self, img, msk):
        """Set Data augmentation.

        Flip image and mask. Rotate image and mask.
        """
        if np.random.rand() > 0.5:
            ax = np.random.choice([0, 1])
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        if np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            img = np.rot90(img, rot, axes=[0, 1])  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=[0, 1])  # Rotate axes 0 and 1

        return img, msk

    def crop_input(self, img, msk):
        """Crop data.

        Randomly crop the image and mask
        """
        slices = []
        # Do we randomize?
        is_random = self.augment and np.random.rand() > 0.5

        for idx, idy in enumerate(range(2)):  # Go through each dimension

            cropLen = self.crop_dim[idx]
            imgLen = img.shape[idy]

            start = (imgLen-cropLen)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + cropLen) > imgLen):
                        # Don't fall off the image
                        start = (imgLen-cropLen)//2
            else:
                start = 0

            slices.append(slice(start, start+cropLen))

        return img[tuple(slices)], msk[tuple(slices)]

    def generate_batch_from_files(self):
        """Generate batch.

        Python generator which goes through a list of filenames to load.
        The files are 3D image (slice is dimension index 2 by default).
        However, we need to yield them as a batch of 2D slices. This generator
        keeps yielding a batch of 2D slices at a time until the 3D image is
        complete and then moves to the next 3D image in the filenames.
        An optional `randomize_slices` allows the user
        to randomize the 3D image slices after loading if desired.
        """
        np.random.seed(self.seed)  # Set a random seed

        idx = 0
        idy = 0

        while True:

            """
            Pack N_IMAGES files at a time to queue
            """
            NUM_QUEUED_IMAGES = 1 + \
                self.batch_size // self.num_slices_per_scan
            # Get enough for full batch + 1
            # print(NUM_QUEUED_IMAGES)
            for idz in range(NUM_QUEUED_IMAGES):

                image_filename = self.filenames[idx]
                label_filename = self.labelnames[idx]

                label = ut.load_mask(label_filename)
                label = dp.preprocess_label(label)

                index_z_crop = []
                if (self.z_slice_min != -1) and (self.z_slice_max != -1):
                    self.imbalanced = False
                    min_ = int(self.z_slice_min*label.shape[0])
                    max_ = int(self.z_slice_max*label.shape[0])
                    index_z_crop = np.arange(min_, max_)
                    label = label[index_z_crop]

                index_imbalanced = []
                if self.imbalanced:
                    for z in range(label.shape[0]):
                        # Check if all 2D numpy array contains only 0
                        result = np.all((label[z] == 0))
                        if not result:
                            index_imbalanced.append(z)
                    index_imbalanced = np.array(index_imbalanced)
                    extra_index = 5
                    index_imbalanced = np.arange(
                        index_imbalanced[0]-extra_index,
                        index_imbalanced[-1]+extra_index)
                    label = label[index_imbalanced]

                while label.shape[0] < self.num_slices_per_scan:
                    stack = self.num_slices_per_scan - label.shape[0]
                    label = np.concatenate((label, label[:stack]), axis=0)

                label = np.moveaxis(label, 0, -1)

                img = ut.load_scan(image_filename)
                img = op.get_pixels_hu(img)

                if (self.z_slice_min != -1) and (self.z_slice_max != -1):
                    img = img[index_z_crop]

                if self.imbalanced:
                    img = img[index_imbalanced]

                while img.shape[0] < self.num_slices_per_scan:
                    stack = self.num_slices_per_scan - img.shape[0]
                    img = np.concatenate((img, img[:stack]), axis=0)

                img = dp.preprocess_img(img)
                img = np.moveaxis(img, 0, -1)

                # Crop input and label
                img, label = self.crop_input(img, label)

                if idz == 0:
                    img_stack = img
                    label_stack = label
                else:
                    img_stack = np.concatenate(
                        (img_stack, img), axis=self.slice_dim)
                    label_stack = np.concatenate(
                        (label_stack, label), axis=self.slice_dim)

                idx += 1
                if idx >= len(self.filenames):
                    idx = 0
                    # Shuffle the filenames/labelnames for the next iteration
                    shuffle = list(zip(self.filenames, self.labelnames))
                    np.random.shuffle(shuffle)
                    self.filenames, self.labelnames = zip(*shuffle)
                    self.filenames = list(self.filenames)
                    self.labelnames = list(self.labelnames)

            img = img_stack
            label = label_stack

            num_slices = img.shape[self.slice_dim]

            if self.batch_size > num_slices:
                raise Exception("Batch size {} is greater than"
                                " the number of slices in the image {}."
                                " Data loader cannot be used.".format(
                                    self.batch_size, num_slices))

            """
            We can also randomize the slices
            so that no 2 runs will return the same slice order
            for a given file.
            This also helps get slices at the end that would be skipped
            if the number of slices is not the same as the batch order.
            """
            if self.augment:
                slice_idx = np.random.choice(range(num_slices), num_slices)
                img = img[:, :, slice_idx]  # Randomize the slices
                label = label[:, :, slice_idx]

            name = self.filenames[idx]

            if (idy + self.batch_size) < num_slices:
                # We have enough slices for batch
                img_batch = img[:, :, idy:idy + self.batch_size]
                label_batch = label[:, :, idy:idy+self.batch_size]
            else:  # We need to pad the batch with slices
                # Get remaining slices
                img_batch = img[:, :, -self.batch_size:]
                label_batch = label[:, :, -self.batch_size:]

            if self.augment:
                img_batch, label_batch = self.augment_data(
                    img_batch, label_batch)

            if len(np.shape(img_batch)) == 3:
                img_batch = np.expand_dims(img_batch, axis=-1)
            if len(np.shape(label_batch)) == 3:
                label_batch = np.expand_dims(label_batch, axis=-1)
            yield np.transpose(img_batch, [2, 0, 1, 3]).astype(np.float32), \
                np.transpose(label_batch, [2, 0, 1, 3]).astype(np.float32)

            idy += self.batch_size
            if idy >= num_slices:  # We finished this file, move to the next
                idy = 0
                idx += 1

            if idx >= len(self.filenames):
                idx = 0
                # Shuffle the filenames/labelnames for the next iteration
                shuffle = list(zip(self.filenames, self.labelnames))
                np.random.shuffle(shuffle)
                self.filenames, self.labelnames = zip(*shuffle)
                self.filenames = list(self.filenames)
                self.labelnames = list(self.labelnames)

    def get_input_shape(self):
        """Get image shape."""
        return [self.crop_dim[0], self.crop_dim[1], 1]

    def get_output_shape(self):
        """Get label shape."""
        return [self.crop_dim[0], self.crop_dim[1], 1]

    def get_dataset(self):
        """Return a dataset."""
        ds = self.generate_batch_from_files()

        return ds

    def __len__(self):
        """Return len."""
        return (self.num_slices_per_scan * self.num_files)//self.batch_size

    def __getitem__(self, idx):
        """Return next item."""
        return next(self.ds)

    def plot_samples(self):
        """Plot some random samples."""
        img, label = next(self.ds)
        print(img.shape)
        plt.figure(figsize=(10, 10))
        slice_num = 2
        plt.subplot(2, 2, 1)
        plt.imshow(img[slice_num, :, :, 0])
        plt.title("Image, Slice #{}".format(slice_num))
        plt.subplot(2, 2, 2)
        plt.imshow(label[slice_num, :, :, 0])
        plt.title("Label, Slice #{}".format(slice_num))
        slice_num = self.batch_size - 1
        plt.subplot(2, 2, 3)
        plt.imshow(img[slice_num, :, :, 0])
        plt.title("Image, Slice #{}".format(slice_num))
        plt.subplot(2, 2, 4)
        plt.imshow(label[slice_num, :, :, 0])
        plt.title("Label, Slice #{}".format(slice_num))
        plt.show()


class DatasetGenerator3D:
    """TensorFlow Dataset from Python/NumPy Iterator."""

    def __init__(self, crop_dim,
                 data_path=None,
                 json_filename=None,
                 batch_size=None,
                 train_test_split=0.8,
                 validate_test_split=0.5,
                 number_output_classes=1,
                 random_seed=None,
                 shard=0):
        """Init function."""
        self.data_path = data_path
        self.json_filename = json_filename
        self.batch_size = batch_size
        self.crop_dim = crop_dim
        self.train_test_split = train_test_split
        self.validate_test_split = validate_test_split
        self.number_output_classes = number_output_classes
        self.random_seed = random_seed
        self.shard = shard  # For Horovod, gives different shard per worker

        self.create_file_list()
        self.ds_train, self.ds_val, self.ds_test = self.get_dataset()

    def create_file_list(self):
        """Create file list."""
        experiment_data = ut.json_export(self.json_filename)
        # Print information about the Magna valve experiment data
        print("*" * 30)
        print("=" * 30)
        print("Dataset name:        ", experiment_data["name"])
        print("Dataset description: ", experiment_data["description"])
        print("Tensor image size:   ", experiment_data["tensorImageSize"])
        print("=" * 30)
        print("*" * 30)
        dataset_folder = self.data_path + experiment_data["dataset_folder"]
        label_folder = self.data_path + experiment_data["label_folder"]
        image_files = ut.expand_list(dataset_folder)
        label_files = ut.expand_list(label_folder)
        self.num_files = len(image_files)
        assert len(image_files) == len(
            label_files), "Files and labels don't have the same length"
        self.filenames = {}
        for idx in range(self.num_files):
            self.filenames[idx] = [image_files[idx], label_files[idx]]

    def z_normalize_img(self, img):
        """Normalize the image.

        Normalize so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[..., channel]
            img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            img[..., channel] = img_temp
        return img

    def crop(self, img, msk, randomize):
        """Randomly crop the image and mask."""
        slices = []
        # Do we randomize?
        is_random = randomize and np.random.rand() > 0.5

        for idx in range(len(img.shape)-1):  # Go through each dimension

            cropLen = self.crop_dim[idx]
            imgLen = img.shape[idx]

            start = (imgLen-cropLen)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    # Don't fall off the image
                    if ((start + cropLen) > imgLen):
                        start = (imgLen-cropLen)//2
            else:
                start = 0

            slices.append(slice(start, start+cropLen))

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk):
        """Get Data augmentation.

        Flip image and mask. Rotate image and mask.
        """
        # Determine if axes are equal and can be rotated
        # If the axes aren't equal then we can't rotate them.
        equal_dim_axis = []
        for idx in range(0, len(self.crop_dim)):
            for jdx in range(idx+1, len(self.crop_dim)):
                if self.crop_dim[idx] == self.crop_dim[jdx]:
                    equal_dim_axis.append([idx, jdx])  # Valid rotation axes
        dim_to_rotate = equal_dim_axis

        if np.random.rand() > 0.5:
            # Random 0,1 (axes to flip)
            ax = np.random.choice(np.arange(len(self.crop_dim)-1))
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        elif (len(dim_to_rotate) > 0) and (np.random.rand() > 0.5):
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            # This will choose the axes to rotate
            # Axes must be equal in size
            random_axis = dim_to_rotate[np.random.choice(len(dim_to_rotate))]
            img = np.rot90(img, rot, axes=random_axis)  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=random_axis)  # Rotate axes 0 and 1
        return img, msk

    def read_files(self, idx, randomize=False):
        """Read dicom and associated labels."""
        idx = idx.numpy()
        img_filename = self.filenames[idx][0]
        label_filename = self.filenames[idx][1]

        img = ut.load_scan(img_filename)
        img = op.get_pixels_hu(img)
        img = np.moveaxis(img, 0, -1)
        img = np.expand_dims(img, -1)

        label = ut.load_mask(label_filename)
        label = dp.preprocess_label(label)
        label = np.moveaxis(label, 0, -1)

        # Combine all masks but background
        if self.number_output_classes == 1:
            label[label > 0] = 1.0
            label = np.expand_dims(label, -1)
        else:
            label_temp = \
                np.zeros(list(label.shape) + [self.number_output_classes])
            for channel in range(self.number_output_classes):
                label_temp[label == channel, channel] = 1.0
            label = label_temp
        # Crop
        img, label = self.crop(img, label, randomize)
        # Normalize
        img = self.z_normalize_img(img)
        # Randomly rotate
        if randomize:
            img, label = self.augment_data(img, label)
        return img, label

    def plot_samples(self, ds, slice_num=1):
        """Plot some random samples."""
        plt.figure(figsize=(20, 20))
        num_cols = 2
        msk_channel = 1
        img_channel = 0
        for img, msk in ds.take(1):
            bs = img.shape[0]
            for idx in range(bs):
                plt.subplot(bs, num_cols, idx*num_cols + 1)
                plt.imshow(img[idx, :, :, slice_num, img_channel], cmap="bone")
                plt.title("MRI", fontsize=18)
                plt.subplot(bs, num_cols, idx*num_cols + 2)
                plt.imshow(msk[idx, :, :, slice_num, msk_channel], cmap="bone")
                plt.title("Tumor", fontsize=18)
        plt.show()
        print("Mean pixel value of image = {}".format(
            np.mean(img[0, :, :, :, 0])))

    def get_train(self):
        """Return train dataset."""
        return self.ds_train

    def get_test(self):
        """Return test dataset."""
        return self.ds_test

    def get_validate(self):
        """Return validation dataset."""
        return self.ds_val

    def get_dataset(self):
        """Create a TensorFlow data loader."""
        self.num_train = int(self.num_files * self.train_test_split)
        num_val_test = self.num_files - self.num_train

        ds = tf.data.Dataset.range(self.num_files).shuffle(
            self.num_files, self.random_seed)  # Shuffle the dataset

        """
        Horovod Sharding
        Here we are not actually dividing the dataset into shards
        but instead just reshuffling the training dataset for every
        shard. Then in the training loop we just go through the training
        dataset but the number of steps is divided by the number of shards.
        """
        ds_train = ds.take(self.num_train).shuffle(
            self.num_train, self.shard)  # Reshuffle based on shard
        ds_val_test = ds.skip(self.num_train)
        self.num_val = int(num_val_test * self.validate_test_split)
        if self.num_val == 0:
            self.num_val = self.num_train
        self.num_test = self.num_train - self.num_val
        if self.num_test == 0:
            self.num_test = self.num_test
        ds_val = ds_val_test.take(self.num_val)
        ds_test = ds_val_test.skip(self.num_val)

        ds_train = ds_train.map(
            lambda x: tf.py_function(self.read_files,
                                     [x, True],
                                     [tf.float32,
                                      tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(
            lambda x: tf.py_function(self.read_files,
                                     [x, False],
                                     [tf.float32, tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(
            lambda x: tf.py_function(self.read_files,
                                     [x, False],
                                     [tf.float32, tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_train = ds_train.repeat()
        ds_train = ds_train.batch(self.batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_val = ds_val.batch(self.batch_size)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.batch(self.batch_size)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_val, ds_test
