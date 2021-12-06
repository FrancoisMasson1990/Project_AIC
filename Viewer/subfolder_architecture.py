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
Redefined folder architecture to have consistency between folders for patient.
"""

import os
from tqdm import tqdm
import shutil
from natsort import natsorted

if __name__ == '__main__':
  
    folder = "R:\Imagerie-Data_Labo\Gr_ClavelMA_img\AIC\datasets_dcm"
    sub_folders = os.listdir(folder)
    for sub_folder in tqdm(sub_folders):
        file_names = list()
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(folder,sub_folder)):
            file_names += [os.path.join(dirpath, file) for file in filenames]
        
        file_names = natsorted(file_names)
        source_dir = file_names[0]
        source_dir = source_dir.split("\\")
        if len(source_dir) > 8 :
            target_dir = source_dir.copy()
            old_dir = "\\".join(source_dir[:7])
            while len(target_dir) > 8:
                target_dir.pop(-3)
            target_dir = "\\".join(target_dir[:-1])
            source_dir = "\\".join(source_dir[:-1])
            shutil.move(source_dir, target_dir)
            shutil.rmtree(old_dir)
        elif len(source_dir) < 8 :
            target_dir = "\\".join(source_dir)[:-4]
            last_digits = int(target_dir.split(".")[-1]) - 1
            target_dir = target_dir.replace(target_dir.split(".")[-1],str(last_digits))
            os.makedirs(target_dir, exist_ok=True)
            for f in file_names:
                shutil.move(f, target_dir)