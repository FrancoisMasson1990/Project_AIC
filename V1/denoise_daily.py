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

if __name__ == "__main__":
    date = datetime.date.today()
    database = "collections.parquet"
    # df = agg.get_collections(database, date)
    df = pd.read_parquet(database)
    columns = ut.get_social_column()
    df = ut.add_social_column(df, database, columns)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # mt.get_twitter_metrics(df, row["twitter"], i)
        mt.get_discord_metrics(df, row["discord"], i)
    print(df["discord_members"])
