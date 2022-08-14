#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Export prediction made by the algo to the
google sheet file online.
"""

import os
import pandas as pd
import numpy as np
import glob
from natsort import natsorted
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm
from aic.misc import files as fs


def get_sheet(url, index=0):
    """Get sheet page."""
    # define the scope
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    json_file = fs.get_valve_credentials()
    # add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_file, scope)
    # authorize the clientsheet
    client = gspread.authorize(creds)
    # get the instance of the Spreadsheet
    sheet = client.open_by_url(url)
    # get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(index)
    return sheet_instance


def get_sheet_cells(df, sheet):
    """Get sheet cells infos."""
    df["patient"] = sheet.col_values(1)
    df["score"] = sheet.col_values(7)
    return df


def update_sheet_cells(df, sheet, col="G", header=4):
    """Update sheet cells infos."""
    start = col + str(header)
    end = col + str(len(df))
    cell_list = sheet.range(start + ":" + end)

    # Write the array to worksheet
    index = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if index >= header - 1:
            cell_list[index].value = row["score"]
            index += 1

    sheet.update_cells(cell_list)


if __name__ == "__main__":

    url = (
        "https://docs.google.com/spreadsheets/d/"
        + "1quuMfuGeg3-DmWE3O8q4sseMFRvOzG0hmKziR4IEpVg/edit?usp=sharing"
    )
    sheet = get_sheet(url)
    df = pd.DataFrame({})
    df = get_sheet_cells(df, sheet)
    df.score = "-"
    folder_datasets = []
    folder_dataset = fs.get_dataset_root()
    for dir in natsorted(os.listdir(folder_dataset)):
        patient = dir.split("/")[0]
        patient = ("-").join(patient.split("-")[:2])
        df.score = np.where(df.patient == patient, "Failed", df.score)

    folder_prediction = fs.get_prediction_root()
    for dir in natsorted(os.listdir(folder_prediction)):
        for sub_dir in natsorted(
            os.listdir(os.path.join(folder_prediction, dir))
        ):
            pkl_file = glob.glob(
                os.path.join(folder_prediction, dir, sub_dir, "*.pkl")
            )[0]
            data = pd.read_pickle(pkl_file)
            patient = data["data_path"].split("/")[0]
            patient = ("-").join(patient.split("-")[:2])
            df.score = np.where(
                df.patient == patient, round(data["score"], 2), df.score
            )

    update_sheet_cells(df, sheet)
