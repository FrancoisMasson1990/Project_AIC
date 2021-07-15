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

import os
import sys

from viewer_3D import Viewer3D
from viewer_2D import Viewer2D
import argparse
import yaml 

import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus and len(sys.argv)> 1 and sys.argv[1].startswith("-a"):
    print("allowing growth")
    growth = True
else:
    print("nogrowth")
    growth = False
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, growth)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
    print(e)

# Deprecated model generated with tf1 version 
# from aic_models import model_2D_old
from aic_models import model_2D

def main_3D(data,folder_mask,folder_npy,multi_label,model,template,model_version):
    viewer = Viewer3D(data,
                      mode=4,
                      label=folder_mask,
                      npy=folder_npy,
                      multi_label=multi_label,
                      model=model,
                      template=template,
                      model_version=model_version)
    viewer.show()

def main_2D(data,folder_mask,model):
    Viewer2D(data,
            folder_mask,
            model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Project AIC')
    arg = parser.parse_args()

    with open('./data_info.yml') as f:
       # The FullLoader parameter handles the conversion from YAML
       # scalar values to Python the dictionary format
       config = yaml.load(f, Loader=yaml.FullLoader)

    arg.data_path = config.get("data_path",None)
    arg.labels_2D = config.get("labels_2D",None)
    arg.folder_mask = config.get("folder_mask",None)
    arg.npy_folder = config.get("npy_folder",None)
    arg.multi_label = config.get("multi_label",None)
    arg.model_name = config.get("model_name",None)
    arg.model_version = config.get("model_version",None)
    arg.template = config.get("template",None)

    # Old version
    #unet_model = model_2D_old.unet()
    unet_model = model_2D.unet()
    if arg.model_name is not None :
        if arg.model_version == 0: 
            model = unet_model.load_model(arg.model_name,False)
        elif arg.model_version == 1:
            model = unet_model.load_model(arg.model_name)
        print("-" * 30)
        print("Model load successfully")
        print("-" * 30)

    sub_folders = os.listdir(arg.data_path)
    data = []
    for sub_folder in sub_folders:
        root = os.path.join(arg.data_path,sub_folder)
        sub_ = os.listdir(root)
        for sub in sub_ :
            data.append(os.path.join(root,sub))

    if arg.labels_2D :
        main_2D(data,
                arg.folder_mask,
                model)
    else :
        main_3D(data,
                arg.folder_mask,
                arg.npy_folder,
                arg.multi_label,
                model,
                arg.template,
                arg.model_version)
