import requests
import pandas as pd
import time 
from opensea import CollectionStats, Collection, Collections

slug = "azuki"
#api = CollectionStats(collection_slug=slug)
#print(api.fetch())
result = Collection(collection_slug=slug)
#print(result.fetch())

#exit()
# headers = None 
# response = requests.request("GET", url, headers=headers, params=querystring)
    
# if "<!doctype html>" in response.text[:20].lower():
#     response = None # blocked request
# else:
#     response = response.json()
#     for collection in response["collections"]:
#         collections.append([collection["name"],collection["slug"]])


url = "https://api.opensea.io/api/v1/collection/azuki"

response = requests.request("GET", url)
response = response.json()

collection_info = response["collection"]
print(collection_info.keys())
print(collection_info["twitter_username"])