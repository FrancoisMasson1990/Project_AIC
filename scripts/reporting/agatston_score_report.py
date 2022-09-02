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
import yaml
from natsort import natsorted
from tqdm import tqdm

import aic.misc.export as expt
import aic.misc.files as fs
from aic.misc.setting_tf import requirements_2d as req2d
from aic.model.inference import get_inference

if __name__ == "__main__":
    req2d()

    config = str(fs.get_configs_root() / "agatston_score_report.yml")
    with open(config) as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)

    """
    Step 0: Get list of valves.
    """

    data_path = config.get("data_path", str(fs.get_dataset_root()))
    model_config = config.get("model_config", None)
    if not model_config:
        raise Exception(f"You should provide a model path.")
    if not os.path.exists(model_config):
        model_config = str(fs.get_configs_root() / model_config)
    save_path = config.get("data_path", str(fs.get_prediction_root()))
    sub_folders = os.listdir(data_path)
    data = []
    for sub_folder in sub_folders:
        root = os.path.join(data_path, sub_folder)
        sub_ = os.listdir(root)
        for sub in sub_:
            if os.path.isdir(os.path.join(root, sub)):
                data.append(os.path.join(root, sub))

    """
    Step 1: Generate prediction from list of valves.
    """

    for d in tqdm(natsorted(data)):
        print(d.split("/")[-2])
        path = os.path.join(save_path, d.split("/")[-2], d.split("/")[-1])
        get_inference(
            data=d,
            config=model_config,
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
            with bz2.BZ2File(prediction, "rb") as f:
                data = pickle.load(f)
            patient = data["data_path"].split("/")[0]
            patient = ("-").join(patient.split("-")[:2])
            df.score = np.where(
                df.patient == patient, round(data["score"], 2), df.score
            )
    df = df[df["patient"].str.contains("AIC")].reset_index()
    expt.update_sheet_cells(df, sheet)
