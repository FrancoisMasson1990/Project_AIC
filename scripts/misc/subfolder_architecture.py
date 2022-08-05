#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Redefined folder architecture to
have consistency between folders for patient.
"""

import os
from tqdm import tqdm
import shutil
from natsort import natsorted


if __name__ == "__main__":

    # Following section give the consistency between subfolder
    # - AIC-###
    #   |
    #   |_ Name Patient
    #       |
    #       |_ dmc files

    folder = "R:\Imagerie-Data_Labo\Gr_ClavelMA_img\AIC\datasets_dcm"
    sub_folders = os.listdir(folder)
    for sub_folder in tqdm(sub_folders):
        file_names = list()
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(folder, sub_folder)):
            file_names += [os.path.join(dirpath, file) for file in filenames]

        file_names = natsorted(file_names)
        source_dir = file_names[0]
        source_dir = source_dir.split("\\")
        if len(source_dir) > 8:
            target_dir = source_dir.copy()
            old_dir = "\\".join(source_dir[:7])
            while len(target_dir) > 8:
                target_dir.pop(-3)
            target_dir = "\\".join(target_dir[:-1])
            source_dir = "\\".join(source_dir[:-1])
            shutil.move(source_dir, target_dir)
            shutil.rmtree(old_dir)
        elif len(source_dir) < 8:
            target_dir = "\\".join(source_dir)[:-4]
            last_digits = int(target_dir.split(".")[-1]) - 1
            target_dir = target_dir.replace(target_dir.split(".")[-1], str(last_digits))
            os.makedirs(target_dir, exist_ok=True)
            for f in file_names:
                shutil.move(f, target_dir)

    # Following section remove -CT1 in folder and files if present
    # data_path = "/home/francoismasson/Project_AIC/valve_patient_folder"
    data_path = "R:/Imagerie-Data_Labo/Gr_ClavelMA_img/AIC"
    surface_label_path = os.path.join(data_path, "labels_2d_npy")
    volume_label_path = os.path.join(data_path, "labels_3d_npy")
    predictions_path = os.path.join(data_path, "predictions")
    data_path = os.path.join(data_path, "datasets_dcm")

    folders = [data_path, surface_label_path, volume_label_path, predictions_path]
    for folder in tqdm(folders):
        for sub in os.listdir(folder):
            if "-CT1" in os.path.join(folder, sub):
                src = os.path.join(folder, sub)
                dst = src.replace("-CT1", "")
                os.rename(src, dst)

            for (dirpath, dirnames, filenames) in os.walk(os.path.join(folder, sub)):
                file_name = [
                    os.path.join(dirpath, file)
                    for file in filenames
                    if file.endswith(".txt")
                ]

            if file_name:
                tmp = "tmp.txt"
                # input file
                fin = open(file_name[0], "rt")
                # output file to write the result to
                fout = open(tmp, "wt")
                # for each line in the input file
                for line in fin:
                    # read replace the string and write to output file
                    fout.write(line.replace("-CT1", ""))
                # close input and output files
                fin.close()
                fout.close()
                shutil.move(tmp, file_name[0])
