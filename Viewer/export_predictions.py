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
Export prediction made by the algo to the
google sheet file online.
"""

import os
import pandas as pd
import numpy as np
import glob
from natsort import natsorted
import gspread
from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm

def get_sheet(url,index=0):
    # define the scope
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    json_file = '/home/francoismasson/Project_AIC/credentials/valve-project-332716-b6570600cfb4.json'
    # add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_file, scope)
    # authorize the clientsheet 
    client = gspread.authorize(creds)
    # get the instance of the Spreadsheet
    sheet = client.open_by_url(url)
    # get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(index)
    return sheet_instance
    
def get_sheet_cells(df,sheet):
    df["patient"] = sheet.col_values(1)
    df["score"] = sheet.col_values(7)
    return df

def update_sheet_cells(df, sheet, col="G"):
    # Write the array to worksheet
    for index,row in tqdm(df.iterrows(),total=len(df)):
        value = row["score"]
        if index >= 3:   
            sheet.update(col+str(index+1), value)

if __name__ == '__main__':
  
    url = "https://docs.google.com/spreadsheets/d/1quuMfuGeg3-DmWE3O8q4sseMFRvOzG0hmKziR4IEpVg/edit?usp=sharing"
    sheet = get_sheet(url)
    df = pd.DataFrame({}) 
    df = get_sheet_cells(df, sheet)
    
    folder_prediction = "/home/francoismasson/Project_AIC/valve_patient_folder/predictions/"
    for dir in natsorted(os.listdir(folder_prediction)):
        for sub_dir in natsorted(os.listdir(os.path.join(folder_prediction,dir))):
            pkl_file = glob.glob(os.path.join(folder_prediction,dir,sub_dir,"*.pkl"))[0]
            data = pd.read_pickle(pkl_file)
            patient = data["data_path"].split("/")[0]
            patient = ("-").join(patient.split("-")[:2])
            df.score = np.where(df.patient == patient,round(data["score"],2),df.score)

    update_sheet_cells(df, sheet)
    