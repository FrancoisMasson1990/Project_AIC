# WIP
import cloudscraper
from bs4 import BeautifulSoup
import json
import pandas as pd

website = "https://crypto.com/price/nft-whale-watch"
website = "https://nftgo.io/whale-tracking/whale"
scraper = cloudscraper.create_scraper() 
headers = {"Accept": "application/json"}
r = scraper.get(website, headers = headers)
soup = BeautifulSoup(r.text, 'html.parser')
data = json.loads(soup.find('script', type='application/json').text)
print(data["props"]["pageProps"].keys())
print(data["props"]["pageProps"]["rankingOverview"])
exit()
df = pd.DataFrame(data["props"]["pageProps"]["nftWhalesResponse"]["data"])
print(df)