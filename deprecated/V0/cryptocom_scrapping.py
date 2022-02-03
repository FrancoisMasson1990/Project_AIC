import pandas as pd
import numpy as np
import requests
import tqdm
import time

# NFT Whale Watch tracks the non-fungible tokens (NFTs) purchased or minted 
# by the largest NFT asset holders on the Ethereum blockchain. 
# ‘Whale’ is a term for individuals or entities that hold large amounts of crypto-related assets.
# This list displays the most recent 50 transactions.

# Need to find other whales sources
headers = {"Accept": "application/json"}

def get_request(url,headers):
    response = requests.request("GET", url, headers=headers)
    response = response.text
    
    return response

def scrapping(response, scrap_mode):
    if scrap_mode == "upcoming":
        delimiter_first = '"pageProps":'
        delimiter_last = '"layoutOptions"'
    elif scrap_mode == "whale":
        delimiter_first = '"pageProps":'
        delimiter_last = '"globalMetrics"'

    try :
        scraper = response.split(delimiter_first)[-1]
        scraper = scraper.split(delimiter_last)[0][:-1] + "}"
        scraper = eval(scraper)
    except:
        scraper = None
    return scraper

whales = []
items = 10 # Number of collections per page
url = "https://crypto.com/price/nft-whale-watch"
scraper = get_request(url, headers)
scrap_mode = "whale"
whale = scrapping(scraper, scrap_mode=scrap_mode)

if whale:
    key = list(whale.keys())[0]
    whale_rows = whale[key]["total"]
    pages = np.ceil(int(whale_rows) / items).astype(int)
    print(f"Found {whale_rows} whales transaction")
else:
    print("No whales transaction")
    exit()


for page in tqdm.tqdm(range(1,pages+1)):
    url = f"{url}/?page={page}"
    scraper = get_request(url, headers)
    whale = scrapping(scraper, scrap_mode=scrap_mode)
    if whale:
        key = list(whale.keys())[0]
        df = pd.DataFrame(whale[key]["data"])
        whales.append(df)
    time.sleep(1)

whales = pd.concat(whales).reset_index(drop=True)

upcomings = []
items = 10 # Number of collections per page
url = "https://crypto.com/price/upcoming-nft"
scraper = get_request(url, headers)
scrap_mode = "upcoming"
upcoming = scrapping(scraper, scrap_mode=scrap_mode)

# Could get key name instead of hardcoding it
if upcoming:
    key = list(upcoming.keys())[0]
    upcomings_rows = upcoming[key]["total"]
    pages = np.ceil(int(upcomings_rows) / items).astype(int)
    print(f"Found {upcomings_rows} upcoming nft collection")
else:
    print("No upcoming collections")
    exit()

for page in tqdm.tqdm(range(1,pages+1)):
    url = f"{url}/?page={page}"
    scraper = get_request(url, headers)
    upcoming = scrapping(scraper, scrap_mode=scrap_mode)
    if upcoming:
        key = list(upcoming.keys())[0]
        df = pd.DataFrame(upcoming[key]["data"])
        upcomings.append(df)
    time.sleep(1)

upcomings = pd.concat(upcomings).reset_index(drop=True)
print(whales)
print(upcomings)
