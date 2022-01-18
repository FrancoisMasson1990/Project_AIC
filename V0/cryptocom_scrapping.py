import cloudscraper
import json 
import itertools 

# variables
url = "https://crypto.com/price/nft-whale-watch?page=2"
url = "https://crypto.com/price/upcoming-nft"


scraper = cloudscraper.create_scraper(browser={'browser': 'chrome',
                                               'platform': 'android',
                                               'desktop': False})

html = scraper.get(url).text
print(html)
