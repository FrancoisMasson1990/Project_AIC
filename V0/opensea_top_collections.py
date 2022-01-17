import pandas as pd
import cloudscraper
import json 
import itertools 

scraper = cloudscraper.create_scraper(browser={'browser': 'chrome',
                                               'platform': 'android',
                                               'desktop': False})
# variables
url = "https://opensea.io/rankings"
volume = ["one_day_volume","seven_day_volume","thirty_day_volume","total_volume"]
categories = [None,"new","art","collectibles","domain-names","music","photography-category",
              "sports","trading-cards","utility","virtual-worlds"] # None = all categories
chains = [None,"ethereum","matic","klaytn"] # None = all chains
categories = ["new","art","collectibles","domain-names","music","photography-category",
              "sports","trading-cards","utility","virtual-worlds"] # None = all categories
chains = ["ethereum","matic","klaytn"] # None = all chains

for element in itertools.product(*[volume, categories, chains]):
    print(element)
    filters = {"sortBy": None, "category": None, "chain": None}
    filters["sortBy"] = element[0]
    filters["category"] = element[1]
    filters["chain"] = element[2]

    remove = []
    for key,value in filters.items():
        if not value:
            remove.append(key)

    for r in remove:
        filters.pop(r)



    html = scraper.get(url,params=filters).text
    html = html.split('"json":')[-1].split(',"pageInfo":')[0] + "}}}"
    json_response = json.loads(html)["data"]["rankings"]["edges"]
    nodes = []
    for node in json_response:
        nodes.append(node["node"])

    df = pd.DataFrame(nodes)
    print(df)
    print(df["stats"])
    exit()