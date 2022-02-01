#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library to aggregate data into Denoise format
from top collections on a daily basis
"""

import pandas as pd
import utils as ut
from tqdm import tqdm
import metrics as mt
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def get_tops():
    df_nftscoring = pd.DataFrame()
    # df_nftscoring = get_nftscoring_data()
    df_opensea = pd.DataFrame()
    df_opensea = get_opensea_data()
    exit()
    df_list = [df_nftscoring, df_opensea]
    dfs = pd.concat(df_list)
    dfs.drop_duplicates(subset=['name'], inplace=True)
    dfs.reset_index(drop=True, inplace=True)
    return dfs


def get_nftscoring_data():
    print("Scrap nftscoring.com...")
    # variables
    url = "https://nftscoring.com/allCollections"
    driver = ut.selenium_driver()
    # Hardcoded values but allows to scrap all collections content
    # These values should change over time
    driver.set_window_size(1000, 150000)
    driver.get(url=url)
    rows_list = []

    x_path = '//*[contains(@class, "table-row")]'
    try:
        elems = \
            WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, x_path)))
    except Exception as e:
        print(e)
        elems = []

    for elem in tqdm(elems):
        row = {"name": None,
               "twitter": None,
               "discord": None,
               "website": None}
        href_path = './/*[@href]'
        hrefs = elem.find_elements(By.XPATH, href_path)
        for href in hrefs:
            if "twitter" in href.get_attribute("href"):
                row["twitter"] = href.get_attribute("href")
            elif "discord" in href.get_attribute("href"):
                row["discord"] = href.get_attribute("href")
            elif "detail" in href.get_attribute("href"):
                temp_page = ut.selenium_driver()
                temp_page.get(url=href.get_attribute("href"))
                x_path = '//*[@id="app"]/section/main/section[1]/section'
                x_path = '//*[contains(@class, "self-start")]'
                extra = \
                    WebDriverWait(temp_page, 10).until(
                        EC.visibility_of_all_elements_located(
                            (By.XPATH, x_path)))
                for e in extra:
                    hrefs_extra = e.find_elements(By.XPATH, href_path)
                    for href_extra in hrefs_extra:
                        if "opensea.io" in href_extra.get_attribute("href"):
                            name = href_extra.get_attribute("href")
                            row["name"] = name.split("/")[-1]
                        elif ("twitter" not in href_extra.get_attribute(
                            "href")) \
                                and ("discord" not in href_extra.get_attribute(
                                    "href")):
                            row["website"] = href_extra.get_attribute("href")
                temp_page.close()
        rows_list.append(row)
    driver.close()
    df = pd.DataFrame(rows_list)
    df.drop_duplicates(subset=['twitter', 'discord'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_opensea_data():
    print("Scrap opensea.io...")
    url = "https://opensea.io/rankings"
    url_filter = "script"
    collection_filter = "edges"
    table_filter = "node"
    slug = "slug"
    scrap = ut.scrap_url(url, url_filter)
    scrap = ut.json_extract(scrap, collection_filter)
    rows_list = []

    if scrap:
        scrap = scrap[0]
        count = len(scrap)
    else:
        raise Exception("Wrong request")

    for i in tqdm(range(count)):
        row = {"name": None,
               "twitter": None,
               "discord": None,
               "website": None}
        row["name"] = scrap[i][table_filter][slug]
        infos = mt.get_opensea_infos(row["name"])
        collection = infos["collection"]
        row["twitter"] = collection.get('twitter_username', None)
        row["discord"] = collection.get('discord_url', None)
        row["website"] = collection.get('external_url', None)
        rows_list.append(row)
    df = pd.DataFrame(rows_list)

    print(df)
    exit()
