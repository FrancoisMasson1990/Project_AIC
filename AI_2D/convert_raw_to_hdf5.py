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

For BraTS (Task 1):

LABEL_CHANNELS: "labels": {
	 "0": "background",
	 "1": "Magna_valve",
}

"""

import os
import nibabel as nib  # pip install nibabel
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

	img = resampling(img,args.resize)
	img = normalize_img(img)

	return img

def preprocess_labels(msk):
	"""
	Process the ground truth labels
	"""

	## Stack the loaded npy files
	msk = [np.load(msk[i]) for i in range(len(msk))]
	msk = np.stack(msk, axis=0)

	if len(msk.shape) != 4:  # Make sure 4D
		msk = np.expand_dims(msk, -1)

	msk = resampling(msk,args.resize)

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

def convert_raw_data_to_hdf5(filename, dataDir, json_data):
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

	# Training filenames / Duplicate for validation/test set

	image_files = expand_list(json_data["dataset_folder"],format='/*.dcm')
	label_files = expand_list(json_data["label_folder"],format='/*.npy')

	attach_attributes(hdf_file, image_files, "input_files")
	attach_attributes(hdf_file, label_files, "label_files")
	
	"""
	Print shapes of raw data
	"""
	print("Data shapes")
	print("===========")
	print("n.b. All tensors converted to stacks of 2D slices.")
	print("If you want true 3D tensors, then modify this code appropriately.")
	
	images = load_scan(image_files[0])
	imgs = get_pixels_hu(images)
	imgs = preprocess_inputs(imgs).shape[1:]
	print("Process Image shape = (?, {}, {}, {})".format(imgs[0],
														 imgs[1],
														 imgs[2]))
	
	msk = load_mask(label_files[0])
	msk = preprocess_labels(msk).shape[1:]
	print("Process Masks shape = (?, {}, {}, {})".format(msk[0],
														 msk[1],
														 msk[2]))

	# Save training set images
	print("Step 1 of 6. Save training set images.")
	first = True
	for idx in tqdm(image_files):

		images = load_scan(idx)
		imgs = get_pixels_hu(images)
		imgs = preprocess_inputs(imgs)
		
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
		else:
			row = img_train_dset.shape[0]  # Count current dataset rows
			img_train_dset.resize(row + num_rows, axis=0)  # Add new row
			# Insert data into new row
			img_train_dset[row:(row + num_rows), :] = imgs

	# Save training set masks
	print("Step 4 of 6. Save training set masks.")
	first = True
	for idx in tqdm(label_files):

		msk = load_mask(idx)
		msk = preprocess_labels(msk)
		num_rows = msk.shape[0]

		if first:
			first = False
			msk_train_dset = hdf_file.create_dataset("msks_train",
													 msk.shape,
													 maxshape=(None,
															   msk.shape[1],
															   msk.shape[2],
															   msk.shape[3]),
													 dtype=float)
			msk_train_dset[:] = msk
		else:
			row = msk_train_dset.shape[0]  # Count current dataset rows
			msk_train_dset.resize(row + num_rows, axis=0)  # Add new row
			# Insert data into new row
			msk_train_dset[row:(row + num_rows), :] = msk

	hdf_file.close()
	print("Finished processing.")
	print("HDF5 saved to {}".format(filename))


if __name__ == "__main__":

	print("Converts the Medical AIC project raw dicom files into" 
			"single HDF5 file for easier use in TensorFlow/Keras.")

	parser = argparse.ArgumentParser(description="Medical AIC project data ",add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--data_path",default=None,help="Path to the datafiles")
	parser.add_argument("--save_path",default=None,help="Folder to save Numpy data files")
	parser.add_argument("--output_filename",default="project_aic.h5",help="Name of the output HDF5 file")
	parser.add_argument("--resize", type=int, default=240,help="Resize height and width to this size. Original size = 240")
	parser.add_argument("--split", type=float, default=0.85,help="Train/test split ratio")

	args = parser.parse_args()

	save_dir = args.save_path

	if save_dir is None:
		save_dir = expanduser("~")

	# Create directory
	try:
		os.makedirs(save_dir)
	except OSError:
		if not os.path.isdir(save_dir):
			raise

	filename = os.path.join(save_dir, args.output_filename)

	# Check for existing output file and delete if exists
	if os.path.exists(filename):
		print("Removing existing data file: {}".format(filename))
		os.remove(filename)


	"""
	Get the training file names from the data directory.
	Decathlon should always have a dataset.json file in the
	subdirectory which lists the experiment information including
	the input and label filenames.
	"""

	json_filename = os.path.join(args.data_path, "dataset.json")

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

	# Set the random seed so that always get same random mix
	np.random.seed(816)

	"""
	Once the dataset fully enough on data, could perform train/val/split.
	Right now only conversion of all data in a single list.
	"""
	
	# numFiles = experiment_data["numTraining"]
	# idxList = np.arange(numFiles)  # List of file indices
	# randomList = np.random.random(numFiles)  # List of random numbers
	
	# Random number go from 0 to 1. So anything above
	# args.train_split is in the validation list.
	# trainList = idxList[randomList < args.split]

	# otherList = idxList[randomList >= args.split]
	# randomList = np.random.random(len(otherList))  # List of random numbers
	# validateList = otherList[randomList >= 0.5]
	# testList = otherList[randomList < 0.5]

	convert_raw_data_to_hdf5(filename, args.data_path,experiment_data)
