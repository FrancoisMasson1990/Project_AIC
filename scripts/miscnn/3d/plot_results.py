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
Copyright (C) 2022 Project AIC - All Rights Reserved.

Print out a visual representation of the inference from a 3D trained model.
"""

import aic.misc.files as fs
import aic.misc.plots as plt

if __name__ == "__main__":

    """
    Load the filepath
    """
    # file_path = "../data_kidney/case_00018/"
    file_path = str(fs.get_dataset_root() / "AIC-002")
    plt.plot_results_3d_miscnn(file_path=file_path)
