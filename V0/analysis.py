# connect to google 
# https://hackernoon.com/how-to-use-google-trends-api-with-python

import os
import pandas as pd
from pytrends.request import TrendReq
import plotly.graph_objects as go
import datetime
import chart_studio
import chart_studio.plotly as py

username = 'francois.masson' # your username
api_key = '6jAq2ymELJmOJWBQ9Tme' # your api key - go to profile > settings > regenerate key in plotlyaccount
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

filter_date = datetime.datetime(2021,1,1)
kw_list = ["cryptopunks"]
dataset = "./dataset"

for dir in os.listdir(dataset):
    asset = dir.lower()
    path = os.path.join(dataset,dir,"number_sales.csv")
    number_sales = pd.read_csv(path)
    number_sales[number_sales.columns[0]] = pd.to_datetime(number_sales[number_sales.columns[0]])
    number_sales = number_sales[number_sales[number_sales.columns[0]]>filter_date]

    path = os.path.join(dataset,dir,"sales_usd.csv")
    sales_usd = pd.read_csv(path)
    sales_usd[sales_usd.columns[0]] = pd.to_datetime(sales_usd[sales_usd.columns[0]])
    sales_usd = sales_usd[sales_usd[sales_usd.columns[0]]>filter_date]

    path = os.path.join(dataset,dir,"price_usd.csv")
    price_usd = pd.read_csv(path)
    price_usd[price_usd.columns[0]] = pd.to_datetime(price_usd[price_usd.columns[0]])
    price_usd = price_usd[price_usd[price_usd.columns[0]]>filter_date]

    assets = [asset,asset+ " nft"]
    trends = TrendReq(hl='en-US', tz=360)
    trends.build_payload(assets, cat=0, timeframe='today 12-m') 
    #1 Interest over Time
    trends_data = trends.interest_over_time()
    trends_data = trends_data.reset_index() 
    trends_data[trends_data.columns[0]] = pd.to_datetime(trends_data[trends_data.columns[0]])
    trends_data = trends_data[trends_data[trends_data.columns[0]]>filter_date]
    trends_data[asset] += trends_data[asset+ " nft"]

    #queries  = pytrends.related_queries()
    #print(queries[kw_list]['top']
    #fig = px.line(time_data, x="date", y=kw_list, title='Keyword Web Search Interest Over Time')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sales_usd[sales_usd.columns[0]],
        y=sales_usd[sales_usd.columns[1]],
        name=sales_usd.columns[1]
    ))

    fig.add_trace(go.Scatter(
        x=number_sales[number_sales.columns[0]],
        y=number_sales[number_sales.columns[1]],
        name=number_sales.columns[1],
        yaxis="y2"
    ))

    fig.add_trace(go.Scatter(
        x=trends_data[trends_data.columns[0]],
        y=trends_data[trends_data.columns[1]],
        name=trends_data.columns[1]+"_queries",
        yaxis="y3"
    ))

    fig.add_trace(go.Scatter(
        x=price_usd[price_usd.columns[0]],
        y=price_usd[price_usd.columns[1]],
        name=price_usd.columns[1],
        yaxis="y4"
    ))

    # Create axis objects
    fig.update_layout(
        xaxis=dict(
            domain=[0.2, 0.7]
        ),
        yaxis=dict(
            title=sales_usd.columns[1]),
        yaxis2=dict(
            title=number_sales.columns[1],
            anchor="free",
            overlaying="y",
            side="left",
            position=0.15
        ),
        yaxis3=dict(
            title=price_usd.columns[1],
            anchor="x",
            overlaying="y",
            side="right"
        ),
        yaxis4=dict(
            title=trends_data.columns[1]+"_queries",
            anchor="free",
            overlaying="y",
            side="right",
            position=0.75
    )
    )

    # Update layout properties
    fig.update_layout(title_text=asset)
    #fig.show()
    py.plot(fig, filename = asset, auto_open=True)