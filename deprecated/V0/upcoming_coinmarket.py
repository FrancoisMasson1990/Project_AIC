import pandas as pd
import numpy as np
import requests
import tqdm
import time

# variables
api_key = "d29ef65d-d1d6-46aa-8339-7202b36eb4f7"
items = 20 # Number of collections per page
url = "https://coinmarketcap.com/nft/upcoming/"
headers = {"Accept": "application/json"}
collections = []

def get_request(url,headers):
    response = requests.request("GET", url, headers=headers)
    response = response.text
    
    return response

def scrapping(response):
    try :
        scraper = response.split('"pageProps":')[-1]
        scraper = scraper.split('"platforms"')[0][:-1] + "}}"
        scraper = eval(scraper)
    except:
        scraper = None
    return scraper

scraper = get_request(url, headers)
upcoming = scrapping(scraper)

if upcoming:
    collections_rows = upcoming["upcomingNFTs"]["count"]
    pages = np.ceil(int(collections_rows) / items).astype(int)
    print(f"Found {collections_rows} upcoming nft collection")
else:
    print("No upcoming collections")
    exit()


for page in tqdm.tqdm(range(1,pages+1)):
    url = f"https://coinmarketcap.com/nft/upcoming/?page={page}"
    scraper = get_request(url, headers)
    upcoming = scrapping(scraper)
    if upcoming:
        df = pd.DataFrame(upcoming["upcomingNFTs"]["upcomings"])
        collections.append(df)
    time.sleep(1)

collections = pd.concat(collections).reset_index(drop=True)
collections.to_csv("upcoming.csv")
collections.to_excel("upcoming.xlsx")