#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Aggregate data into a single database coming from both
upcoming and top collections on a daily basis
"""

import pandas as pd
import top_aggregator as top
import upcoming_aggregator as up
import os
import sql as sql
import utils as ut
import metrics as mt
from tqdm import tqdm


def get_collections(new, old, date):
    # Step 1 : Gather new collection
    upcomings_df = up.get_upcomings()
    # Step 2 : Gather top collection
    top_df = top.get_tops()
    df = pd.concat([upcomings_df, top_df])
    df.reset_index(drop=True, inplace=True)
    df.insert(loc=0,
              column="date",
              value=date.strftime('%Y-%m-%d'))

    # Step 2 : Update database from the day before
    if os.path.exists(old):
        new_collections = []
        # check if collection already present or not
        # append if new info
        df_collections = sql.load_sql(old)
        for i, row in upcomings_df.iterrows():
            if len(df_collections[df_collections["name"] == row["name"]]) == 0:
                new_collections.append(row)
        if new_collections:
            new_collections = pd.concat(new_collections, axis=1).T
            df_collections = pd.concat([df_collections, new_collections])
            df_collections.reset_index(drop=True, inplace=True)
            sql.to_sql(df_collections, sql_path=new)
    else:
        sql.to_sql(df, sql_path=new)

    return df


def update_collections(df, name):
    columns = mt.get_social_column()
    columns += mt.get_market_column()
    df = mt.add_column(df, name, columns)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        mt.get_opensea_metrics(df, row["name"], i)
        mt.get_twitter_metrics(df, row["twitter"], row["date"], i)
        mt.get_discord_metrics(df, row["discord"], i)
        mt.get_google_trends(df, row["name"], row["date"], i)
    sql.to_sql(df, sql_path=name)
    return df
