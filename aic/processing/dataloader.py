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

from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
from aic.processing import preprocess as dp


class DatasetGenerator(Sequence):
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
        img = dp.load_scan(filenames[0])
        img = dp.get_pixels_hu(img)
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

            for idz in range(NUM_QUEUED_IMAGES):

                image_filename = self.filenames[idx]
                label_filename = self.labelnames[idx]

                label = dp.load_mask(label_filename)
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

                img = dp.load_scan(image_filename)
                img = dp.get_pixels_hu(img)

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
                img_batch, label_batch = \
                    img[:, :, idy:idy + self.batch_size],
                label[:, :, idy:idy+self.batch_size]

            else:  # We need to pad the batch with slices

                # Get remaining slices
                img_batch, label_batch = \
                    img[:, :, -self.batch_size:],
                label[:, :, -self.batch_size:]

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
