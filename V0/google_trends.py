# connect to google 
# https://hackernoon.com/how-to-use-google-trends-api-with-python

import datetime
import os
from pytrends.request import TrendReq
import pandas as pd

filter_date = datetime.datetime(2021,1,1)
dataset = "./dataset"

for dir in os.listdir(dataset):
    asset = dir.lower()
    assets = [asset,asset+ " nft"]
    if not os.path.exists(os.path.join(dataset,dir,"google_queries.csv")):
        print(asset)
        trends = TrendReq(hl='en-US', tz=360)
        trends.build_payload(assets, cat=0, timeframe='today 12-m') 
        #1 Interest over Time
        trends_data = trends.interest_over_time()
        trends_data = trends_data.reset_index() 
        trends_data[trends_data.columns[0]] = pd.to_datetime(trends_data[trends_data.columns[0]])
        trends_data = trends_data[trends_data[trends_data.columns[0]]>filter_date]
        if len(trends_data)>0:
            trends_data[asset] += trends_data[asset+ " nft"]
            trends_data.to_csv(os.path.join(dataset,dir,"google_queries.csv"),index=False)
        else:
            print(asset)