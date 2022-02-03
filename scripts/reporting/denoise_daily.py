#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Generate daily denoise data by checking new collections
and assign social metrics to them
"""

import denoise.misc.sql as sql
import denoise.aggregator.collections as colc
import denoise.misc.files as fs
import datetime
import os


if __name__ == "__main__":
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    data_folder = fs.get_data_root()
    database_new = \
        data_folder / f'collections_{today.strftime("%y%m%d")}.db'
    database_old = \
        data_folder / f'collections_{yesterday.strftime("%y%m%d")}.db'

    # TODO :
    # Generate temp file to reduce time if crashed and need to rerun the code
    # Optimize code by checking if not already present (twitter/discord filter)
    # for new and top collections
    # Get stats and update values of top collection in merged database
    # Find a way to track minted collection

    if not os.path.exists(database_new):
        df = colc.get_collections(new=database_new,
                                  old=database_old,
                                  date=today)
    else:
        df = sql.load_sql(database_new)
        df = colc.update_collections(df,
                                     database_new)
