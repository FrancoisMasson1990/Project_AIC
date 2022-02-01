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

import pandas as pd
import sql as sql
import aggregator as agg
import datetime
import os


if __name__ == "__main__":
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    database_new = f'collections_{today.strftime("%y%m%d")}.db'
    database_old = f'collections_{yesterday.strftime("%y%m%d")}.db'

    # TODO :
    # Define uniform database schema for upcoming and top collection
    # if concat, missing columns will be set to NaN
    # Scrap opensea to get name, twitter and discord links
    # Optimize code by checking if not already present (twitter/discord filter)
    # for new and top collections
    # Combine upcoming and top collection single database in an efficient way
    # Get stats and update values of top collection in merged database
    # Find a way to track minted collection

    if not os.path.exists(database_new):
        df = agg.get_collections(new=database_new,
                                 old=database_old,
                                 date=today)
    else:
        df = sql.load_sql(database_new)
        df = agg.update_collections(df,
                                    database_new)
