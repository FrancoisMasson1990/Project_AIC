# Get list of all collections available on Opensea
# As per my observation it only retrieves the collections 
# which stats like Volume or 24 H Change are all 0. 
# And the OpenSea rankings requests to the GraphQL server of the OpenSea, 
# which is not available for public.

# https://www.reddit.com/r/opensea/comments/prv9n6/opensea_api_collection/

import requests
import re
import json
import pandas as pd 
import datetime

today = datetime.date.today().strftime('%Y%m%d')

url = "https://api.opensea.io/api/v1/collection/"

offset = 0
limit = 300
opensea_key = "30c5c8bf273f42918ef967bf3c1b1af3"
headers = {"X-API-KEY": opensea_key}

querystring = {"slug":"doodles-official"}

response = requests.request("GET", url, headers=headers, params=querystring)
if response.status_code != 200:
    print('error')


#Getting nft infos
collections = response.json()
