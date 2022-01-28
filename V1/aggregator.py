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
import datetime
import os


if __name__ == "__main__":
    date = datetime.date.today()
    upcomings_df = up.get_upcomings()
    upcomings_df.insert(loc=0,
                        column="date",
                        value=date.strftime('%Y-%m-%d'))
    top_df = pd.DataFrame({})

    df = pd.concat([upcomings_df, top_df])
    df.reset_index(drop=True, inplace=True)

    collections_db = "collections.parquet"
    if os.path.exists(collections_db):
        new_collections = []
        # check if collection present or not
        # append if new info
        df_collections = pd.read_parquet(collections_db)
        for i, row in upcomings_df.iterrows():
            if len(df_collections[df_collections["name"] == row["name"]]) == 0:
                new_collections.append(row)
        if new_collections:
            new_collections = pd.concat(new_collections, axis=1).T
            df_collections = pd.concat([df_collections, new_collections])
            df_collections.reset_index(drop=True, inplace=True)
            df_collections.to_parquet(collections_db)
    else:
        df.to_parquet(collections_db)
