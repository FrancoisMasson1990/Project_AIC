import os
import pandas as pd
import numpy as np
import datetime

filter_date = datetime.datetime(2021,1,1)
dataset = "./dataset"

for dir in os.listdir(dataset):
    asset = dir.lower()
    assets = [asset,asset+ " nft"]

    if os.path.exists(os.path.join(dataset,dir,"google_queries.csv")):
        number_sales = pd.read_csv(os.path.join(dataset,dir,"number_sales.csv"))
        number_sales[number_sales.columns[0]] = pd.to_datetime(number_sales[number_sales.columns[0]])
        number_sales = number_sales[number_sales[number_sales.columns[0]]>filter_date]

        sales_usd = pd.read_csv(os.path.join(dataset,dir,"sales_usd.csv"))
        sales_usd[sales_usd.columns[0]] = pd.to_datetime(sales_usd[sales_usd.columns[0]])
        sales_usd = sales_usd[sales_usd[sales_usd.columns[0]]>filter_date]

        price_usd = pd.read_csv(os.path.join(dataset,dir,"price_usd.csv"))
        price_usd[price_usd.columns[0]] = pd.to_datetime(price_usd[price_usd.columns[0]])
        price_usd = price_usd[price_usd[price_usd.columns[0]]>filter_date]

        # join on datetime
        dataset_agg = number_sales.merge(sales_usd,on='Date').merge(price_usd,on='Date')

        trends_data = pd.DataFrame()
        trends_data[dataset_agg.columns[0]] = dataset_agg[dataset_agg.columns[0]]
        trends_data[asset] = np.NaN

        google_queries = pd.read_csv(os.path.join(dataset,dir,"google_queries.csv"))
        google_queries[google_queries.columns[0]] = pd.to_datetime(google_queries[google_queries.columns[0]])

        for idx,row in google_queries.iterrows():
            trends_data[asset] = \
                np.where((trends_data[dataset_agg.columns[0]] == row["date"]), row[asset], trends_data[asset])
        
        trends_data[asset] = trends_data[asset].interpolate(method='linear',limit_direction='forward')
        dataset_agg = dataset_agg.rename(columns={"Date":"date"})
        dataset_agg[asset] = trends_data[asset]

        dataset_agg = dataset_agg.dropna()
        dataset_agg.to_csv(os.path.join(dataset,dir,"dataset.csv"),index=False)
