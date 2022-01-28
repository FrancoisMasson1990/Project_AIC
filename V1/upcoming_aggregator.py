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
import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def get_upcomings():
    # df_coinmarket = get_coinmarket_data()
    # df_upcoming = get_upcomingnft_data()
    df_nftgo = get_nftgo_data()
    # Must combine info and remove duplicate if present
    df = pd.DataFrame()
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
    df = pd.DataFrame()
    url = "https://upcomingnft.net/upcoming-events/"
    driver = ut.selenium_driver()
    driver.get(url=url)
    rows_list = []

    for i in tqdm(range(1, 2)):
        row = {"name": None,
               "twitter": None,
               "discord": None,
               "website": None,
               "timestamp": None,
               "platform": None,
               "volume": None,
               "mintPrice": None}
        x_path = f'//*[@id="movietable"]/tbody/tr[{i}]'
        elems = \
            WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, x_path)))
        time.sleep(1)
        for elem in elems:
            hrefs = elem.find_elements(By.XPATH, './/*[@href]')
            for href in hrefs:
                if "twitter" in href.get_attribute("href"):
                    row["twitter"] = href.get_attribute("href")
                elif "discord" in href.get_attribute("href"):
                    row["discord"] = href.get_attribute("href")
                elif "upcomingnft.net" in href.get_attribute("href"):
                    name = href.get_attribute("href").split("/")
                    j = -1
                    while not name[j]:
                        j -= 1
                    name = href.get_attribute("href").split("/")[j].split("-")
                    name = " ".join([n.capitalize() for n in name])
                    row["name"] = name
                elif "javascript" not in href.get_attribute("href"):
                    row["website"] = href.get_attribute("href")
            dates = elem.find_elements(By.XPATH, './/*[@class="sorting_1"]')
            for date in dates:
                date = "".join(date.text.split(" "))
                element = datetime.datetime.strptime(date, "%d%b%Y")
                timestamp = datetime.datetime.timestamp(element)
                row["timestamp"] = timestamp
            alts = elem.find_elements(By.XPATH, './/*[@alt]')
            for alt in alts:
                if alt.get_attribute("alt") == "ETH":
                    row["platform"] = "Ethereum"

            volume_path = f'//*[@id="movietable"]/tbody/tr[{i}]/td[7]'
            row["volume"] = ut.get_text(elem, volume_path)
            price_path = f'//*[@id="movietable"]/tbody/tr[{i}]/td[6]'
            row["mintPrice"] = ut.get_text(elem, price_path)
            rows_list.append(row)
        # x_path = '//*[@id="movietable_next"]'
        # element = driver.find_element(By.XPATH, x_path)
        # if element:
        #     element.click()
    df = pd.DataFrame(rows_list)
    print(df)
    driver.close()
    exit()


def get_nftgo_data():
    df = pd.DataFrame()
    url = "https://nftgo.io/nft-drops/"
    driver = ut.selenium_driver()
    driver.get(url=url)
    rows_list = []

    i = 1
    while True:
        row = {"name": None,
               "twitter": None,
               "discord": None,
               "website": None,
               "timestamp": None,
               "platform": None,
               "volume": None,
               "mintPrice": None}

        x_path = f'//*[@id="layout"]/div[2]/div[3]/div[{i}]'
        try:
            elems = \
                WebDriverWait(driver, 10).until(
                    EC.visibility_of_all_elements_located((By.XPATH, x_path)))
            time.sleep(1)
        except Exception as e:
            elems = []
        if not elems:
            break
        for elem in elems:
            hrefs = elem.find_elements(By.XPATH, './/*[@href]')
            for href in hrefs:
                print(href.get_attribute("href"))
                if "twitter" in href.get_attribute("href"):
                    row["twitter"] = href.get_attribute("href")
                elif "discord" in href.get_attribute("href"):
                    row["discord"] = href.get_attribute("href")
                else:
                    row["website"] = href.get_attribute("href")

            name_path = \
                f'//*[@id="layout"]/div[2]/div[3]/div[{i}]/div[1]/div[2]/div[1]'
            row["name"] = ut.get_text(elem, name_path)
            price_path = \
                f'//*[@id="layout"]/div[2]/div[3]/div[{i}]/div[4]/div[1]/div[2]'
            mint_price = ut.get_text(elem, price_path)
            if mint_price:
                row["mintPrice"] = mint_price.split(" ")[0]
                if mint_price.split(" ")[-1] == "ETH":
                    row["platform"] = "Ethereum"
                elif mint_price.split(" ")[-1] == "SQL":
                    row["platform"] = "Solana"
            volume_path = \
                f'//*[@id="layout"]/div[2]/div[3]/div[{i}]/div[4]/div[2]/div[2]'
            row["volume"] = ut.get_text(elem, volume_path)
            # timestamp format conversion
            date_path = \
                f'//*[@id="layout"]/div[2]/div[3]/div[{i}]/div[4]/div[3]/div[2]'
            row["date"] = ut.get_text(elem, date_path)
            rows_list.append(row)
        break
        i += 1
    df = pd.DataFrame(rows_list)
    print(df)
    driver.close()
