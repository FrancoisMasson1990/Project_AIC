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
    return df


def update_sheet_cells(df, sheet, col="G", header=4):
    """Update sheet cells infos."""
    start = col + str(header)
    stop = header + len(df) - 1
    end = col + str(stop)
    cell_list = sheet.range(start + ":" + end)
    # Write the array to worksheet
    index = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        cell_list[index].value = row["score"]
    sheet.update_cells(cell_list)
