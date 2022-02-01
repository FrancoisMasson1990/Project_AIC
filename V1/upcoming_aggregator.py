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
from tqdm import tqdm
import time
import re
import datetime
from dateutil.parser import parse
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def get_upcomings():
    df_coinmarket = pd.DataFrame()
    df_coinmarket = get_coinmarket_data()
    df_upcoming = pd.DataFrame()
    df_upcoming = get_upcomingnft_data()
    df_nftgo = pd.DataFrame()
    df_nftgo = get_nftgo_data()
    df_list = [df_coinmarket, df_upcoming, df_nftgo]
    dfs = pd.concat(df_list)
    dfs.drop_duplicates(subset=['twitter', 'discord'], inplace=True)
    dfs.reset_index(drop=True, inplace=True)
    return dfs


def get_coinmarket_data():
    print("Scrap coinmarketcap.com...")
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
    else:
        raise Exception("Wrong request")

    for page in tqdm(range(1, pages+1)):
        url_ = f"{url}?page={page}"
        scrap = ut.scrap_url(url_, "script")
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
               'volume',
               'timestamp']
    dfs = dfs[columns]
    temp = dfs["mintPrice"].str.split(" ", n=1, expand=True)[0]
    dfs["mintPrice"] = temp
    dfs = dfs[~dfs["mintPrice"].str.isalpha()]
    return dfs


def get_upcomingnft_data():
    print("Scrap upcomingnft.net...")
    df = pd.DataFrame()
    url = "https://upcomingnft.net/upcoming-events/"
    driver = ut.selenium_driver()
    driver.get(url=url)
    rows_list = []
    names = []

    page_info = '//*[@id="movietable_info"]'
    page_info = driver.find_element(By.XPATH, page_info)
    page_info = page_info.get_attribute("innerHTML")
    num = re.findall(r'\d+', page_info)
    while not num:
        print("Error scraping... Retry")
        time.sleep(1)
        page_info = driver.find_element(By.XPATH, page_info)
        page_info = page_info.get_attribute("innerHTML")
        num = re.findall(r'\d+', page_info)
    rows_by_page = int(num[1])
    rows_tot = int(num[-1])
    pages = rows_tot // rows_by_page

    div = rows_tot % rows_by_page
    print(pages)
    if div > 0:
        pages += 1
    for page in tqdm(range(1, pages+1)):
        print(f"page {page}")
        if (page > 1) and (page < pages - 1):
            print("Next page...")
            next_path = '//*[@id="movietable_next"]'
            next_element = \
                driver.find_element(By.XPATH, next_path)
            driver.execute_script("arguments[0].click();",
                                  next_element)
            time.sleep(1)

        if page < pages:
            rows_count = rows_by_page
        elif (page == pages) and (div > 0):
            rows_count = div

        paths = ["'odd'", "'even'"]
        for path in paths:
            if path == "'odd'":
                i = 1
            elif path == "'even'":
                i = 2
            x_path = f'.//*[contains(@class, {path})]'
            x_path = x_path.replace("'", '"')
            elems = driver.find_elements(By.XPATH, x_path)
            time.sleep(1)
            for elem in elems:
                row = {"name": None,
                       "twitter": None,
                       "discord": None,
                       "website": None,
                       "timestamp": None,
                       "platform": None,
                       "volume": None,
                       "mintPrice": None}

                hrefs = elem.find_elements(By.XPATH, './/*[@href]')
                for href in hrefs:
                    if "twitter" in href.get_attribute("href"):
                        row["twitter"] = href.get_attribute("href")
                    elif "discord" in href.get_attribute("href"):
                        row["discord"] = href.get_attribute("href")
                    elif "upcomingnft.net/event" in href.get_attribute("href"):
                        name = href.get_attribute("href").split("/")
                        j = -1
                        while not name[j]:
                            j -= 1
                        name = \
                            href.get_attribute("href").split("/")[j].split("-")
                        name = " ".join([n.capitalize() for n in name])
                        row["name"] = name
                    elif "javascript" not in href.get_attribute("href"):
                        row["website"] = href.get_attribute("href")

                dates = \
                    elem.find_elements(By.XPATH, './/*[@class="sorting_1"]')
                for date in dates:
                    date = date.get_attribute("innerHTML")
                    date = parse(date)
                    timestamp = datetime.datetime.timestamp(date)
                    row["timestamp"] = timestamp
                alts = elem.find_elements(By.XPATH, './/*[@alt]')
                for alt in alts:
                    if alt.get_attribute("alt") == "ETH":
                        row["platform"] = "Ethereum"

                volume_path = f'//*[@id="movietable"]/tbody/tr[{i}]/td[7]'
                volumes = \
                    elem.find_elements(By.XPATH, volume_path)
                for volume in volumes:
                    volume = volume.get_attribute("innerHTML").split("> ")[-1]
                    row["volume"] = volume
                price_path = f'//*[@id="movietable"]/tbody/tr[{i}]/td[6]'
                mints = \
                    elem.find_elements(By.XPATH, price_path)
                for mint in mints:
                    mint = mint.get_attribute("innerHTML").split("> ")[-1]
                    row["mintPrice"] = mint

                if row["name"] not in names:
                    rows_list.append(row)
                    names.append(name)
                    i += 2

    df = pd.DataFrame(rows_list)
    driver.close()
    return df


def get_nftgo_data():
    print("Scrap nftgo.io...")
    df = pd.DataFrame()
    url = "https://nftgo.io/nft-drops/"
    driver = ut.selenium_driver()
    driver.get(url=url)
    rows_list = []

    x_path = '//*[contains(@class, "nft-drop-card_card")]'
    try:
        elems = \
            WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, x_path)))
        if elems:
            for elem in tqdm(elems):
                row = {"name": None,
                       "twitter": None,
                       "discord": None,
                       "website": None,
                       "timestamp": None,
                       "platform": None,
                       "volume": None,
                       "mintPrice": None}
                hrefs = elem.find_elements(By.XPATH, './/*[@href]')
                for href in hrefs:
                    if "twitter" in href.get_attribute("href"):
                        row["twitter"] = href.get_attribute("href")
                    elif "discord" in href.get_attribute("href"):
                        row["discord"] = href.get_attribute("href")
                    else:
                        row["website"] = href.get_attribute("href")
                name_path = './/*[contains(@class, "nft-drop-card_name")]'
                row["name"] = ut.get_text(elem, name_path)
                stats_path = './/*[contains(@class, "nft-drop-card_stats")]'
                stats = ut.get_text(elem, stats_path)
                stats = stats.split("\n")
                mint_price = stats[1]
                row["mintPrice"] = mint_price.split(" ")[0]
                if mint_price.split(" ")[-1] == "ETH":
                    row["platform"] = "Ethereum"
                elif (mint_price.split(" ")[-1] == "SOL") \
                        or mint_price.split(" ")[-1] == "Solana":
                    row["platform"] = "Solana"
                elif mint_price.split(" ")[-1] == "MATIC":
                    row["platform"] = "Matic"
                if stats[3]:
                    row["volume"] = float(stats[3].replace(",", ""))
                if stats[5]:
                    date = parse(stats[5])
                    timestamp = datetime.datetime.timestamp(date)
                    row["timestamp"] = timestamp
                rows_list.append(row)
    except Exception as e:
        print(e)

    driver.close()
    df = pd.DataFrame(rows_list)
    df.drop_duplicates(subset=['twitter', 'discord'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
