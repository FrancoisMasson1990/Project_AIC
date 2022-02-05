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
import denoise.misc.files as fs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
headers = {"Accept": "application/json"}


def get_request(url, headers):
    """Get url request."""
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


def scrap_url(url, key="script", params={}, property_=None):
    """Scrap url html infos."""
    scraper = cloudscraper.create_scraper()
    r = scraper.get(url, params=params, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    if property_:
        values = soup.find(key, property=property_)
        values = values.get("content", None)
    else:
        values = soup.find(key, type='application/json')
        if values:
            values = json.loads(values.text)
        else:
            values = None
    return values


def selenium_driver():
    """Get Selenium Service."""
    path = "../chromedriver/chromedriver"
    browserpath = "/opt/google/chrome/google-chrome"

    options = webdriver.ChromeOptions()
    options.binary_location = browserpath
    options.add_argument('--headless')

    service = Service(path)
    driver = uc.Chrome(service=service, options=options)
    return driver


def get_text(elem, x_path):
    """Get text info using selenium properties."""
    value = elem.find_elements(By.XPATH, x_path)
    if value:
        return value[0].text
    else:
        return None


def get_twitter_keys(name="twitter_key.json"):
    """Get twitter keys access."""
    root = fs.get_keys_root()
    keys = root / name
    return keys
