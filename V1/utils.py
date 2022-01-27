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


def scrap_url(url, key="script"):
    scraper = cloudscraper.create_scraper()
    r = scraper.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    json_data = json.loads(soup.find(key, type='application/json').text)
    return json_data
