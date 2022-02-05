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
import denoise.aggregator.top_aggregator as top
import denoise.aggregator.upcoming_aggregator as up
import denoise.misc.sql as sql
import denoise.misc.metrics as mt
import denoise.misc.files as fs
from tqdm import tqdm
import os


def get_collections(new, old, date):
    """Get collection of NFT."""
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
        for name in up.get_upcoming_names():
            db_temp = fs.get_data_root() / (name + ".db")
            if os.path.exists(db_temp):
                os.remove(db_temp)
        for name in top.get_top_names():
            db_temp = fs.get_data_root() / (name + ".db")
            if os.path.exists(db_temp):
                os.remove(db_temp)
    return df


def update_collections(df,
                       name,
                       opensea_metrics=[],
                       discord_metrics=[],
                       google_metrics=[],
                       twitter_metrics=[],
                       ):
    """Update collection infos of NFT."""
    columns = mt.get_social_column()
    columns += mt.get_market_column()
    df = mt.add_column(df, name, columns)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        discord_metrics.append(
            mt.get_discord_metrics(row["discord"]))
        opensea_metrics.append(
            mt.get_opensea_metrics(row["name"]))
        google_metrics.append(
            mt.get_google_trends(row["name"]))
        twitter_metrics.append(
            mt.get_twitter_metrics(row["twitter"],
                                   row["date"]))

    df[mt.get_discord_column()] = discord_metrics
    df[mt.get_google_column()] = google_metrics
    df[mt.get_twitter_column()] = twitter_metrics
    df[mt.get_opensea_column()] = opensea_metrics
    sql.to_sql(df, sql_path=name)
    return df
