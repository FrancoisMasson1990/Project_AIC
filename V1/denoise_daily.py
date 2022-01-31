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
import aggregator as agg
import utils as ut
import datetime
import metrics as mt
from tqdm import tqdm
import os

if __name__ == "__main__":
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    database_new = f'collections_{today.strftime("%y%m%d")}.parquet'
    database_old = f'collections_{yesterday.strftime("%y%m%d")}.parquet'

    # TODO :
    # Define uniform database schema for upcoming and top collection
    # Scrap nftscoring to get name, twitter and discord links
    # Need to mimick a scrolling attributes
    # Combine upcoming and top collection single database in an efficient way
    # Get stats and update values of top collection in merged database
    # Find a way to track minted collection
    # Optimize code by checking if not already present (twitter/discord filter)
    # for new and top collections

    import upcoming_aggregator as up
    up.get_nftscoring_collection()
    exit()

    if not os.path.exists(database_new):
        df = agg.get_collections(new=database_new,
                                 old=database_old,
                                 date=today)
    else:
        df = pd.read_parquet(database_new)
        columns = ut.get_social_column()
        df = ut.add_social_column(df, database_new, columns)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            mt.get_twitter_metrics(df, row["twitter"], row["date"], i)
            mt.get_discord_metrics(df, row["discord"], i)
            mt.get_google_trends(df, row["name"], row["date"], i)
        df.to_parquet(database_new)

    # How do I track minted collection ?
