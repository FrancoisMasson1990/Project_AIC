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
Converts the Medical AIC project raw dicom files into
single HDF5 file for easier use in TensorFlow/Keras.

Dataset is split in two parts : data_preprocess & label_mask
So far not using .zip files but may be an option later

For Medical AIC (Task 1):

LABEL_CHANNELS: "labels": {
	 "0": "background",
	 "1": "Magna_valve",
}

"""

import os
import numpy as np
from tqdm import tqdm  # pip install tqdm
import h5py   # pip install h5py
import json
from os.path import expanduser
import argparse
import glob
import pydicom
from skimage.transform import resize
from natsort import natsorted
import copy
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator


LABEL_CHANNELS = {"labels":{
	 			  "background":0,
				  "other":1,
	 			  "Magna_valve":2,
				 }}

def normalize_img(img):
	"""
	Normalize the pixel values.
	This is one of the most important preprocessing steps.
	We need to make sure that the pixel values have a mean of 0
	and a standard deviation of 1 to help the model to train
	faster and more accurately.
	"""

	for channel in range(img.shape[3]):
		img[:, :, :, channel] = (
			img[:, :, :, channel] - np.mean(img[:, :, :, channel])) \
			/ np.std(img[:, :, :, channel])

	return img

def crop_center(img, cropx, cropy, cropz):
	"""
	Take a center crop of the images.
	If we are using a 2D model, then we'll just stack the
	z dimension.
	"""
	z, x, y, c = img.shape

	# Make sure starting index is >= 0
	startx = max(x // 2 - (cropx // 2), 0)
	starty = max(y // 2 - (cropy // 2), 0)

	# Make sure ending index is <= size
	endx = min(startx + cropx, x)
	endy = min(starty + cropy, y)

	return img[:, startx:endx, starty:endy, :]

def resampling(img,size):

	new_image = np.empty((img.shape[0],size,size,1))

	for i in range(img.shape[0]):
		new_image[i] = resize(img[i], (size,size))

	return new_image

def attach_attributes(df, json_data, name):
	"""
	Save the json data
	"""

	if type(json_data) is str:
		length = 1
	else:
		length = len(json_data)

	dt = h5py.special_dtype(vlen=str)
	dset = df.create_dataset(name, (length,), dtype=dt)
	dset[:] = json_data


def preprocess_inputs(img):
	"""
	Process the input images
	"""
	if len(img.shape) != 4:  # Make sure 4D
		img = np.expand_dims(img, -1)

	if (args.resize != -1):
		img = crop_center(img, args.resize, args.resize, -1)
	img = normalize_img(img)

	return img

def preprocess_labels(msk):
	"""
	Process the ground truth labels
	"""

	## Stack the loaded npy files
	msk = [np.load(msk[i]) for i in range(len(msk))]
	msk = np.stack(msk, axis=0)

	if intel_model :
		if len(msk.shape) != 4:  # Make sure 4D
			msk = np.expand_dims(msk, -1)
	else :
		# extract certain classes from mask
		msks = [(msk == v) for v in LABEL_CHANNELS["labels"].values()]
		msk = np.stack(msks, axis=-1).astype('float')
    	
	# Cropping
	if (args.resize != -1):
		msk = crop_center(msk, args.resize, args.resize, -1)
	
	return msk

def expand_list(data_path, format):

	sub_folders = os.listdir(data_path)
	data = []
	for sub_folder in sub_folders:
		root = os.path.join(data_path, sub_folder)
		sub_ = os.listdir(root)
		for sub in sub_:
			data.append(os.path.join(root, sub))

	data = natsorted(data)

	return data


def load_scan(path):
	files = os.listdir(path)
	file_dcm = []
	for f in files:
		if f.endswith('.dcm'):
			file_dcm.append(f)
	slices = [pydicom.read_file(path + '/' + s) for s in file_dcm]
	slices.sort(key = lambda x: int(x.InstanceNumber))
	
	try:
		slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
	
	for s in slices:
		s.SliceThickness = slice_thickness
	
	return slices

def load_mask(path):	
	mask = glob.glob(path + '/*.npy')
	mask = natsorted(mask)

	return mask

def get_pixels_hu(scans):
	image = np.stack([s.pixel_array for s in scans])
	image = image.astype(np.int16)
	image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
	intercept = scans[0].RescaleIntercept
	slope = scans[0].RescaleSlope

	if slope != 1:
		image = slope * image.astype(np.float64)
		image = image.astype(np.int16)
		
	image += np.int16(intercept)

	return np.array(image, dtype=np.int16)

def test_train_val_split(image_files,split):
	"""
	Once the dataset fully enough on data, could perform train/val/split.
	Right now only conversion of all data in a single list.
	"""
	# Set the random seed so that always get same random mix
	np.random.seed(42)
	idxList = np.arange(len(image_files))  # List of file indices
	randomList = np.random.random(len(image_files))  # List of random numbers
	#Random number go from 0 to 1. So anything above
	#split is in the validation list.
	trainList = idxList[randomList < split]
	otherList = idxList[randomList >= split]
	randomList = np.random.random(len(otherList))  # List of random numbers

	validateList = otherList[randomList >= 0.5]
	testList = otherList[randomList < 0.5]
	
	return trainList,validateList,testList 

def imbalanced_data_counter(image,msks):
	"""
	Get a repartition of the ratio of the different classes.Go through the dataset.json file.
	This done image wise and pixel wise
	"""
	# Pixel Wise 
	total_pixel = image.shape[0] * image.shape[1] * image.shape[2] * image.shape[3] 

	print("\n")
	for key,value in LABEL_CHANNELS["labels"].items():
		count = (msks[:,:,:,0] == value).sum()
		ratio = 100*count/total_pixel
		print("pixel wise ratio (%) of {} is {}".format(key,str(ratio)))

	#Image Wise
	for key,value in LABEL_CHANNELS["labels"].items():
		count = 0
		for l in range(msks.shape[0]):
			if value == 0 :
				is_value = np.all((msks[l,:,:,0] == value))
			else :
				is_value = np.any((msks[l,:,:,0] == value))
			if is_value :
				count += 1 
		print("image wise ratio (%) of {} is {}".format(key,str(count/msks.shape[0])))
	print("\n")

def imbalanced_data_augmentation(imgs,msks,total=20,seed=42):
	# construct the image generator for data augmentation then
	# initialize the total number of images generated thus far
	aug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode="nearest")
	
	# First find masks with label 1. Ignore the full 0
	msks_stack = []
	index = []
	for i in range(msks.shape[0]):
		msks_ = msks[i,:,:,:]
		is_value = np.any((msks_ != 0))
		if is_value:
			index.append(i)
			msks_ = np.expand_dims(msks_, 0)
			# prepare iterator
			it = aug.flow(msks_, batch_size=1,seed=seed)
			# generate samples
			for i in range(total):
				batch = it.next()
				msks_stack.append(batch)

	index = np.array(index)
	imgs_augmented = imgs[index,:,:,:]

	imgs_stack = []
	for i in range(imgs_augmented.shape[0]):
		imgs_ = imgs_augmented[i,:,:,:]
		imgs_ = np.expand_dims(imgs_, 0)
		# prepare iterator
		it = aug.flow(imgs_, batch_size=1,seed=seed)
		# generate samples
		for i in range(total):
			batch = it.next()
			imgs_stack.append(batch)
	
	imgs_augmented = np.vstack(imgs_stack)
	msks_augmented = np.vstack(msks_stack)

	#Add the non augmented imgs/mask
	imgs_augmented = np.vstack((imgs[~index,:,:,:],imgs_augmented))
	msks_augmented = np.vstack((msks[~index,:,:,:],msks_augmented))
	
	return imgs_augmented,msks_augmented

def convert_raw_data_to_hdf5(filename, dataDir, json_data, split):
	"""
	Go through the dataset.json file.
	We've already split into training and validation subsets.
	Read in Dicom format files. Crop images and masks.
	Save to HDF5 format.
	This code is will convert the 3D images and masks
	into a stack of 2D slices.
	"""

	hdf_file = h5py.File(filename, "w")

	# Save the dataset attributes
	attach_attributes(hdf_file, json_data["name"], "name")
	attach_attributes(hdf_file, json_data["description"], "description")
	attach_attributes(hdf_file, json_data["release"], "release")
	attach_attributes(hdf_file, json_data["tensorImageSize"], "tensorImageSize")

	image_files = expand_list(json_data["dataset_folder"],format='/*.dcm')
	label_files = expand_list(json_data["label_folder"],format='/*.npy')

	assert len(image_files) == len(label_files), "Files and labels don't have the same length"
	
	image_files = np.asarray(image_files)
	label_files = np.asarray(label_files)

	# Test/train/val split
	train_list_index,val_list_index,test_list_index = test_train_val_split(image_files,split)
	train_list_index = np.asarray(train_list_index)
	val_list_index = np.asarray(val_list_index)
	test_list_index = np.asarray(test_list_index)

	# Training filenames
	train_image_files = image_files[train_list_index].tolist()
	train_label_files = label_files[train_list_index].tolist()

	# Validation filenames
	validate_image_files = image_files[val_list_index].tolist()
	validate_label_files = label_files[val_list_index].tolist()

	# Testing filenames
	test_image_files = image_files[test_list_index].tolist()
	test_label_files = label_files[test_list_index].tolist()

	attach_attributes(hdf_file, train_image_files, "training_input_files")
	attach_attributes(hdf_file, train_label_files, "training_label_files")
	attach_attributes(hdf_file, validate_image_files, "validation_input_files")
	attach_attributes(hdf_file, validate_label_files, "validation_label_files")
	attach_attributes(hdf_file, test_image_files, "testing_input_files")
	attach_attributes(hdf_file, test_label_files, "testing_label_files")

	image_files = image_files.tolist()
	label_files = label_files.tolist()

	"""
	Print shapes of raw data
	"""
	print("Data shapes")
	print("===========")
	print("n.b. All tensors converted to stacks of 2D slices.")
	print("If you want true 3D tensors, then modify this code appropriately.")
	
	images = load_scan(image_files[0])
	imgs = get_pixels_hu(images)
	imgs_ = preprocess_inputs(imgs).shape[1:]
	print("Process Image shape = (?, {}, {}, {})".format(imgs_[0],
														 imgs_[1],
														 imgs_[2]))
	
	msk = load_mask(label_files[0])
	msk_ = preprocess_labels(msk).shape[1:]
	print("Process Masks shape = (?, {}, {}, {})".format(msk_[0],
														 msk_[1],
														 msk_[2]))

	# Print out the ratio of one exemple

	imbalanced_data_counter(preprocess_inputs(imgs),preprocess_labels(msk))

	# Save training set images
	print("Step 1 of 3. Save training set images and masking set images.")
	first = True
	for idx,idx_ in tqdm(zip(train_image_files,train_label_files)):

		images = load_scan(idx)
		imgs = get_pixels_hu(images)
		imgs = preprocess_inputs(imgs)

		msk = load_mask(idx_)
		msk = preprocess_labels(msk)
		
		assert msk.shape[0] == imgs.shape[0]

		# Data augmentation for the training part
		#imgs,msk = imbalanced_data_augmentation(imgs,msk)
		num_rows = imgs.shape[0]

		if first:
			first = False
			img_train_dset = hdf_file.create_dataset("imgs_train",
													 imgs.shape,
													 maxshape=(None,
															   imgs.shape[1],
															   imgs.shape[2],
															   imgs.shape[3]),
													 dtype=float)
			img_train_dset[:] = imgs

			msk_train_dset = hdf_file.create_dataset("msks_train",
													 msk.shape,
													 maxshape=(None,
															   msk.shape[1],
															   msk.shape[2],
															   msk.shape[3]),
													 dtype=float)
			msk_train_dset[:] = msk

		else:
			row = img_train_dset.shape[0]  # Count current dataset rows
			img_train_dset.resize(row + num_rows, axis=0)  # Add new row
			# Insert data into new row
			img_train_dset[row:(row + num_rows), :] = imgs

			row = msk_train_dset.shape[0]  # Count current dataset rows
			msk_train_dset.resize(row + num_rows, axis=0)  # Add new row
			# Insert data into new row
			msk_train_dset[row:(row + num_rows), :] = msk

	# Save validation set images
	print("Step 2 of 3. Save validation set images and masking set images.")
	first = True
	for idx,idx_ in tqdm(zip(validate_image_files,validate_label_files)):

		images = load_scan(idx)
		imgs = get_pixels_hu(images)
		imgs = preprocess_inputs(imgs)
		
		msk = load_mask(idx_)
		msk = preprocess_labels(msk)
		
		assert msk.shape[0] == imgs.shape[0]
		# Data augmentation for the validation part
		# Required also because need to force model to improve for low imbalanced data in the val set
		#imgs,msk = imbalanced_data_augmentation(imgs,msk)
		num_rows = imgs.shape[0]

		if first:
			first = False
			img_validation_dset = hdf_file.create_dataset("imgs_validation",
													 imgs.shape,
													 maxshape=(None,
															   imgs.shape[1],
															   imgs.shape[2],
															   imgs.shape[3]),
													 dtype=float)
			img_validation_dset[:] = imgs

			msk_validation_dset = hdf_file.create_dataset("msks_validation",
													 msk.shape,
													 maxshape=(None,
															   msk.shape[1],
															   msk.shape[2],
															   msk.shape[3]),
													 dtype=float)
			msk_validation_dset[:] = msk

		else:
			row = img_validation_dset.shape[0]  # Count current dataset rows
			img_validation_dset.resize(row + num_rows, axis=0)  # Add new row
			# Insert data into new row
			img_validation_dset[row:(row + num_rows), :] = imgs

			row = msk_validation_dset.shape[0]  # Count current dataset rows
			msk_validation_dset.resize(row + num_rows, axis=0)  # Add new row
			# Insert data into new row
			msk_validation_dset[row:(row + num_rows), :] = msk

	# Save testing set images
	print("Step 3 of 3. Save testing set images and masking set images.")
	first = True
	for idx,idx_ in tqdm(zip(test_image_files,test_label_files)):

		images = load_scan(idx)
		imgs = get_pixels_hu(images)
		imgs = preprocess_inputs(imgs)
		
		msk = load_mask(idx_)
		msk = preprocess_labels(msk)
		
		assert msk.shape[0] == imgs.shape[0]
		num_rows = imgs.shape[0]

		if first:
			first = False
			img_testing_dset = hdf_file.create_dataset("imgs_testing",
													 imgs.shape,
													 maxshape=(None,
															   imgs.shape[1],
															   imgs.shape[2],
															   imgs.shape[3]),
													 dtype=float)
			img_testing_dset[:] = imgs

			msk_testing_dset = hdf_file.create_dataset("msks_testing",
													 msk.shape,
													 maxshape=(None,
															   msk.shape[1],
															   msk.shape[2],
															   msk.shape[3]),
													 dtype=float)
			msk_testing_dset[:] = msk

		else:
			row = img_testing_dset.shape[0]  # Count current dataset rows
			img_testing_dset.resize(row + num_rows, axis=0)  # Add new row
			# Insert data into new row
			img_testing_dset[row:(row + num_rows), :] = imgs

			row = msk_testing_dset.shape[0]  # Count current dataset rows
			msk_testing_dset.resize(row + num_rows, axis=0)  # Add new row
			# Insert data into new row
			msk_testing_dset[row:(row + num_rows), :] = msk

	hdf_file.close()
	print("Finished processing.")
	print("HDF5 saved to {}".format(filename))


if __name__ == "__main__":

	print("Converts the Medical AIC project raw dicom files into" 
			"single HDF5 file for easier use in TensorFlow/Keras.")

	parser = argparse.ArgumentParser(description="Medical AIC project data ",add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--resize", type=int, default=-1,help="Resize height and width to this size. Original size = 512")
	parser.add_argument("--split", type=float, default=0.5,help="Train/test split ratio")

	args = parser.parse_args()

	with open('./preprocess_data.yml') as f:
		# The FullLoader parameter handles the conversion from YAML
		# scalar values to Python the dictionary format
		config = yaml.load(f, Loader=yaml.FullLoader)

	data_path = config.get("data_path",None)
	output_filename = config.get("output_filename",None)
	save_dir = config.get("save_path",None)
	intel_model = config.get("intel_model",None)

	if save_dir is None:
		save_dir = expanduser("~")

	# Create directory
	try:
		os.makedirs(save_dir)
	except OSError:
		if not os.path.isdir(save_dir):
			raise

	filename = os.path.join(save_dir, output_filename)
		
	# Check for existing output file and delete if exists
	if os.path.exists(filename):
		print("Removing existing data file: {}".format(filename))
		os.remove(filename)

	"""
	All the useful infos are stored in a json file (dataset.json)
	"""

	json_filename = os.path.join(data_path, "dataset.json")

	try:
		with open(json_filename, "r") as fp:
			experiment_data = json.load(fp)
	except IOError as e:
		print("File {} doesn't exist. It should be part of the "
			  "Dataset directory".format(json_filename))

	# Print information about the Decathlon experiment data
	print("*" * 30)
	print("=" * 30)
	print("Dataset name:        ", experiment_data["name"])
	print("Dataset description: ", experiment_data["description"])
	print("Tensor image size:   ", experiment_data["tensorImageSize"])
	print("Dataset release:     ", experiment_data["release"])
	print("Dataset location:     ", experiment_data["dataset_folder"])
	print("Label location:     ", experiment_data["label_folder"])
	print("=" * 30)
	print("*" * 30)

	"""
	Grap folder of data and train/val/split
	and randomize the file list.
	"""

	convert_raw_data_to_hdf5(filename,data_path,experiment_data,args.split)
