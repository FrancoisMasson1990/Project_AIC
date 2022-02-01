#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Miscellaneous function used in Denoise project
"""
import requests
import cloudscraper
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
headers = {"Accept": "application/json"}


def get_request(url, headers):
    response = requests.request("GET", url, headers=headers)
    response = response.text
    return response


def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    arr.append(v)
                elif isinstance(v, (dict, list)):
                    extract(v, arr, key)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values


def scrap_url(url, key="script", property=None):
    scraper = cloudscraper.create_scraper()
    r = scraper.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    if property:
        values = soup.find(key, property=property)
        values = values.get("content", None)
    else:
        values = json.loads(soup.find(key, type='application/json').text)
    return values


def selenium_driver():
    path = "../chromedriver/chromedriver"
    browserpath = "/opt/google/chrome/google-chrome"

    options = webdriver.ChromeOptions()
    options.binary_location = browserpath
    options.add_argument('--headless')

    service = Service(path)
    driver = uc.Chrome(service=service, options=options)
    return driver


def get_text(elem, x_path):
    value = elem.find_elements(By.XPATH, x_path)
    print(value[0].get_attribute("innerHTML"))
    exit()
    if value:
        print(value[0].get_attribute("innerHTML"))
        return value[0].text
    else:
        return None


def get_social_column():
    columns = ["google_trend",
               "twitter_followers"
               "twitter_post",
               "twitter_retweet",
               "twitter_like",
               "discord_members"]
    return columns


def add_social_column(df, name, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = 0
        df.to_pickle(name)
    return df
