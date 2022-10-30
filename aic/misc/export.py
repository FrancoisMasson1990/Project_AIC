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
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

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


def get_sheet_cells(sheet, range_cell: str = ""):
    """Get sheet cells infos."""
    return pd.DataFrame(sheet.get(range_cell))


def update_sheet_cells(df, sheet):
    """Update sheet cells infos from a dataframe."""
    sheet.update(df.values.tolist())
