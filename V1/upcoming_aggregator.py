#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library to aggregate data into Denoise format
from upcoming collections on a daily basis
"""

import pandas as pd
import numpy as np
import utils as ut
import tqdm
import time


def get_upcomings():
    df_coinmarket = get_coinmarket_data()
    df_upcoming = get_upcomingnft_data()
    # Must combine info and remove duplicate if present

    return df


def get_coinmarket_data():
    dfs = []
    url = "https://coinmarketcap.com/nft/upcoming/"
    url_filter = "script"
    collection_filter = "upcomingNFTs"
    table_filter = "upcomings"
    scrap = ut.scrap_url(url, url_filter)
    scrap = ut.json_extract(scrap, collection_filter)
    if scrap:
        scrap = scrap[0]
        count = scrap["count"]
        items = len(scrap["upcomings"])
        pages = np.ceil(int(count) / items).astype(int)
        print(f"Found {count} upcoming nft collection")
    else:
        raise Exception("Wrong request")

    for page in tqdm.tqdm(range(1, pages+1)):
        url = f"{url}?page={page}"
        scrap = ut.scrap_url(url, "script")
        scrap = ut.json_extract(scrap, "upcomingNFTs")
        if scrap:
            df = pd.DataFrame(scrap[0][table_filter])
            dfs.append(df)
            time.sleep(1)
        else:
            raise Exception("Wrong request")
    dfs = pd.concat(dfs)
    dfs.reset_index(drop=True, inplace=True)
    columns = ['name',
               'twitter',
               'discord',
               'website',
               'platform',
               'mintPrice',
               'presalePrice',
               'volume',
               'timestamp']
    dfs = dfs[columns]
    # Should rename columns
    return dfs


def get_upcomingnft_data():
    pass
