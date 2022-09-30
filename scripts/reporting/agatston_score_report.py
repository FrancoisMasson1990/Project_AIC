#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Generate automatically prediction and export to google sheet.
"""

import bz2
import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

import aic.misc.export as expt
import aic.misc.files as fs
import aic.model.loaders as ld
from aic.misc.setting_tf import requirements_2d as req2d
from aic.model.inference import get_inference

if __name__ == "__main__":
    req2d()

    config_path = str(fs.get_configs_root() / "agatston_score_report.yml")
    config = ld.load_config(config_path)

    """
    Step 0: Get list of valves.
    """

    data_path = config.get("data_path", str(fs.get_dataset_root()))
    model_config_path = config.get("model_config", None)
    if not model_config_path:
        raise Exception(f"You should provide a model path.")
    if not os.path.exists(model_config_path):
        model_config_path = str(fs.get_configs_root() / model_config_path)

    save_path = config.get("save_path", str(fs.get_prediction_root()))
    model_config = ld.load_config(model_config_path)
    model_name = model_config.get("model_name", None)
    online = model_config.get("online", False)
    if model_name:
        if online:
            model = ld.load_tflitemodel(model_name)
        else:
            model = ld.load_model(model_name)
    else:
        model = None

    sub_folders = os.listdir(data_path)
    data = []
    for sub_folder in natsorted(sub_folders):
        root = os.path.join(data_path, sub_folder)
        sub_ = os.listdir(root)
        for sub in sub_:
            if os.path.isdir(os.path.join(root, sub)):
                data.append(os.path.join(root, sub))

    """
    Step 1: Generate prediction from list of valves.
    """

    batch_start = 0
    batch_stop = -1  # -1 for last element
    except_list = ["AIC-015"]
    for d in tqdm(data[batch_start:batch_stop]):
        print(d.split("/")[-2])
        if d.split("/")[-2] in except_list:
            continue
        path = os.path.join(save_path, d.split("/")[-2], d.split("/")[-1])
        get_inference(
            data=d,
            config=model_config_path,
            model=model,
            online=False,
            lite=False,
            save_path=path,
        )

    """
    Step 2: Export prediction score to Google Sheet.
    """

    url = config.get("url", None)
    sheet = expt.get_sheet(url)
    df = pd.DataFrame({})
    df = expt.get_sheet_cells(df, sheet)
    df["score"] = "-"
    folder_datasets = []
    folder_dataset = fs.get_dataset_root()
    for dir in natsorted(os.listdir(folder_dataset)):
        patient = dir.split("/")[0]
        patient = ("-").join(patient.split("-")[:2])
        df.score = np.where(df.patient == patient, "In Progress", df.score)

    folder_prediction = fs.get_prediction_root()
    for dir in natsorted(os.listdir(folder_prediction)):
        for sub_dir in natsorted(
            os.listdir(os.path.join(folder_prediction, dir))
        ):
            prediction = glob(
                os.path.join(folder_prediction, dir, sub_dir, "*.pbz2")
            )[0]
            if prediction:
                with bz2.BZ2File(prediction, "rb") as f:
                    data = pickle.load(f)
                patient = data["data_path"].split("/")[0]
                patient = ("-").join(patient.split("-")[:2])
                df.score = np.where(
                    df.patient == patient, round(data["score"], 2), df.score
                )
    df = df[df["patient"].str.contains("AIC")].reset_index()
    expt.update_sheet_cells(df, sheet)
