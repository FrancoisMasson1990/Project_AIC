import requests
import pandas as pd
from tqdm import tqdm
import time 

from opensea import Collections
api = Collections().fetch().items
print(api)
exit()

df = pd.read_pickle("collection.pkl")
#print(df)
# print(df[df.Name.str.contains("clonex")])
# exit()
slug = "clonex"
from opensea import CollectionStats, Collection
api = CollectionStats(collection_slug=slug)
print(api.fetch())
result = Collection(collection_slug=slug)
print(result.fetch())
exit()

MAX_API_ITEMS = 300
MAX_OFFSET = 10000
MAX_ITERATIONS = MAX_OFFSET/MAX_API_ITEMS

i = 0
rate_limiting = 2
collections = []
while i < MAX_ITERATIONS:
    # rate limiting
    if rate_limiting != None:
        time.sleep(rate_limiting)
    print("---\nFetching transactions from OpenSea...")
    print("{}/{}".format(i,MAX_ITERATIONS))

    url = "https://api.opensea.io/api/v1/collections"

    limit=MAX_API_ITEMS, 
    offset=i*MAX_API_ITEMS, 

    querystring = {"offset":"{}".format(offset[0]), 
                   "limit":"{}".format(limit[0]),
                   }

    headers = None 
    response = requests.request("GET", url, headers=headers, params=querystring)
    
    if "<!doctype html>" in response.text[:20].lower():
        response = None # blocked request
    else:
        response = response.json()
        for collection in response["collections"]:
            collections.append([collection["name"],collection["slug"]])

    # iterate
    i += 1

# {'offset': ['ensure this value is less than or equal to 50000']}
# collections = []
# for offset in tqdm(range(0,5000,300)):
#     url = f"https://api.opensea.io/api/v1/collections?offset={offset}&limit=300"
#     response = requests.request("GET", url)
#     response = response.json()
#     for collection in response["collections"]:
#         collections.append([collection["name"],collection["slug"]])

df = pd.DataFrame(data=collections, columns=["Name","Slug"])
df.to_pickle("collection.pkl")