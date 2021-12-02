from numpy.core.defchararray import title
import pandas as pd
from prophet import Prophet
import datetime
#from prophet.plot import plot_plotly
import numpy as np
import os
import sys
sys.path.append("/home/francoismasson/time_series/")
from plot_utils import plot_plotly

import chart_studio
import chart_studio.plotly as py

username = 'francois.masson' # your username
api_key = '6jAq2ymELJmOJWBQ9Tme' # your api key - go to profile > settings > regenerate key in plotlyaccount
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv").iloc[:, :2] 
dataset = "./dataset"
nfts = ["CryptoPunks",
        "Bored Ape Yacht Club",
        "Art Blocks",
        "Meebits",
        "Loot",
        "SuperRare",
        "CyberKongz",
        "Cool Cats",
        "MekaVerse",
        "VeeFriends",
        "0n1 Force",
        "Pudgy Penguins",
        "Foundation",
        "The Sandbox",
        "Hashmasks",
        "Decentraland",
        "The Doge Pound",
        "Sorare",
        "Doodles",
        ]

# date for futur prediction
start_date = datetime.date(2021,9,3)
end_date = datetime.date(2021,11,3)
periods = end_date - start_date

show = False
for dir in os.listdir(dataset):
    if dir in nfts:
        asset = dir.lower()
        if os.path.exists(os.path.join(dataset,dir,"dataset.csv")):
            print(dir)
            df = pd.read_csv(os.path.join(dataset,dir,"dataset.csv"))
            columns = df.columns
            index = 3
            df = df[[columns[0],columns[index]]]
            df = df.rename(columns={columns[0]: 'ds', columns[index]: 'y'})

            if len(df[df.ds<start_date.strftime("%Y-%m-%d")])>0:
                m = Prophet()
                m.fit(df[df.ds<start_date.strftime("%Y-%m-%d")])

                date_future = m.make_future_dataframe(periods=periods.days)
                df_forecast = m.predict(date_future)
                df_forecast["real"] = df.y
                df_forecast["real"] = \
                    np.where((df_forecast.ds < start_date.strftime("%Y-%m-%d")), np.NaN, df_forecast.real)
                df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'real']]

                fig = plot_plotly(m, df_forecast, xlabel='Time', ylabel='Price USD') 
                # Update layout properties
                title = "Price prediction of {} using time series method".format(asset.capitalize())
                fig.update_layout(title_text=title)
                if show:
                    fig.show()
                else:
                    py.plot(fig, filename = asset + "_time_series", auto_open=True)