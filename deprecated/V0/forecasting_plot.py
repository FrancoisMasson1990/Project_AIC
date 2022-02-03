import datetime
import tweepy as tw
import json
import pandas as pd
import numpy as np
import os
from prophet import Prophet
from tqdm import tqdm
import plotly.graph_objects as go
import chart_studio
import chart_studio.plotly as py

username = 'francois.masson' # your username
api_key = '6jAq2ymELJmOJWBQ9Tme' # your api key - go to profile > settings > regenerate key in plotlyaccount
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    
if __name__ == "__main__":

    # Documentation for tweepy can be found here
    # https://docs.tweepy.org/en/stable/api.html

    twitter_keys = "twitter_key.json"
    with open(twitter_keys) as json_file:
        keys = json.load(json_file)
    
    auth = tw.AppAuthHandler(keys["api_key"], keys["api_key_secret"])
    api = tw.API(auth,wait_on_rate_limit=True)

    collections_name = ["CryptoPunks",
                        "Bored Ape Yacht Club",
                        #"Art Blocks",
                        #"Meebits",
                        #"Loot",
                        #"SuperRare",
                        "CyberKongz",
                        #"Cool Cats",
                        #"MekaVerse",
                        #"VeeFriends",
                        #"0n1 Force",
                        #"Pudgy Penguins",
                        #"Foundation",
                        #"The Sandbox",
                        #"Hashmasks",
                        #"Decentraland",
                        #"The Doge Pound",
                        #"Sorare",
                        #"Doodles",
                        ]

    path = "./tweets/"

    with open(path + 'nft_influencers.txt') as f:
        influencers = f.readlines()
    
    collect = False
    df_collections = []
    for influencer in tqdm(influencers):
        userID = influencer.split('\n')[0]
        if userID.startswith("#"):
            continue
        table = path + '%s_tweets.csv' % userID
        oldest_id = None

        if os.path.exists(table):
            df = pd.read_csv(table)
            oldest_id = df.id.iloc[-1]

        if collect:
            if oldest_id is not None:
                tweets = api.user_timeline(screen_name=userID, 
                                           # 200 is the maximum allowed count
                                           count=200,
                                           include_rts = False,
                                           max_id = oldest_id - 1,
                                           # Necessary to keep full_text 
                                           # otherwise only the first 140 words are extracted
                                           tweet_mode = 'extended'
                                           )
            else:
                tweets = api.user_timeline(screen_name=userID, 
                                           # 200 is the maximum allowed count
                                           count=200,
                                           include_rts = False,
                                           # Necessary to keep full_text 
                                           # otherwise only the first 140 words are extracted
                                           tweet_mode = 'extended'
                                           )
            all_tweets = []
            all_tweets.extend(tweets)
            if len(tweets)>0:
                oldest_id = tweets[-1].id
            else:
                print("No (more) tweets found")
                continue
            while True:
                tweets = api.user_timeline(screen_name=userID, 
                                           # 200 is the maximum allowed count
                                           count=200,
                                           include_rts = False,
                                           max_id = oldest_id - 1,
                                           # Necessary to keep full_text 
                                           # otherwise only the first 140 words are extracted
                                           tweet_mode = 'extended'
                                           )
                if len(tweets) == 0:
                    break
                oldest_id = tweets[-1].id
                all_tweets.extend(tweets)
                print('N of tweets downloaded till now {}'.format(len(all_tweets)))
            
            outtweets = [[tweet.id_str, 
                        tweet.created_at, 
                        tweet.favorite_count, 
                        tweet.retweet_count, 
                        tweet.full_text.encode("utf-8").decode("utf-8")] 
                        for idx,tweet in enumerate(all_tweets)]
            df = pd.DataFrame(outtweets,columns=["id","created_at","favorite_count","retweet_count", "text"])

            if not os.path.exists(table):
                # write to a csv file
                df.to_csv(table,index=False)
            else : 
                # append lines to csv files
                df.to_csv(table,index=False, mode = 'a', header=False)
    
        else:
            df = pd.read_csv(table)
            # keywords = ["Bored"]
            # keywords = "|".join(keywords)
            for collection in collections_name:
                collection_split = collection.split(" ")
                if "The" in collection_split:
                    collection_split.remove("The")
                keywords = [*collection_split]
                keywords = "|".join(keywords)        
                df_collection = df[df.text.str.contains(keywords,case = False, regex = True)].copy()
                if len(df_collection)> 0:
                    df_collection["collection"] = collection
                    df_collection["userID"] = userID
                df_collections.append(df_collection)

    df_collections = pd.concat(df_collections)
    df_collections["time"] = pd.to_datetime(df_collections["created_at"]).dt.date
    df_collections = df_collections[df_collections.time > datetime.date(2021,1,1)]
    df_collections = df_collections[["time","collection","favorite_count","retweet_count"]]
    df_collections = df_collections.groupby(["time","collection"]).sum()
    df_collections = pd.DataFrame(df_collections.to_records())
    df_collections = df_collections.sort_values(by=["time"])

    param = "favorite_count"
    #param = "retweet_count"
    target = 'Average USD (7-day)'

    for i,collection_name in enumerate(collections_name):
        df_collections_name = df_collections[df_collections.collection == collection_name]
        df_dataset = pd.read_csv(f"./dataset/{collection_name}/dataset.csv")

        fig = go.Figure()

        timeline = len(df_dataset)
        init_time = 60
        period = 7

        # Invisible graphe
        fig.add_trace(go.Scatter(
            x=df_dataset["date"],
            y=df_dataset["Average USD (7-day)"],
            name="Price USD",
            line=dict(color='rgba(255,255,255,0)'),
            visible=True,
            showlegend=False),
            )
        
        fig.add_trace(go.Scatter(
            x=df_dataset["date"].iloc[:init_time+1],
            y=df_dataset["Average USD (7-day)"].iloc[:init_time+1],
            name="Price USD",
            line=dict(color='#2ca02c'),
            visible=True),
            )

        df_prophet = df_dataset.rename(columns={'date': 'ds', 'Average USD (7-day)': 'y'})
        m = Prophet()
        m.fit(df_prophet.iloc[:init_time])
        date_future = m.make_future_dataframe(periods=period)
        df_forecast = m.predict(date_future)
        df_forecast = df_forecast.iloc[-period:]

        if df_forecast['yhat'].iloc[0] < df_dataset["Average USD (7-day)"].iloc[init_time]:
            corr_factor = np.abs(df_dataset["Average USD (7-day)"].iloc[init_time] - df_forecast['yhat'].iloc[0])
        else:
            corr_factor = - np.abs(df_dataset["Average USD (7-day)"].iloc[init_time] - df_forecast['yhat'].iloc[0])

        # Add lower bound
        fig.add_trace(go.Scatter(
            name="Price USD",
            x=df_forecast['ds'],
            y=df_forecast['yhat_lower']+corr_factor,
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip',
            visible=True,
            showlegend=False),
            )

        fig.add_trace(go.Scatter(
            name='Forecast prediction',
            x=df_forecast['ds'],
            y=df_forecast['yhat']+corr_factor,
            mode='lines',
            line=dict(color='firebrick'),
            fillcolor='rgba(255, 255, 255, 0.2)',
            fill='tonexty',
            visible=True,
            showlegend=True),
            )

        # Add upper bound
        fig.add_trace(go.Scatter(
            x=df_forecast['ds'],
            y=df_forecast['yhat_upper']+corr_factor,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 255, 255, 0.2)',
            fill='tonexty',
            hoverinfo='skip',
            visible=True,
            showlegend=False),
            )

        for t in range(1,len(df_dataset)//period):
            fig.add_trace(go.Scatter(
                x=df_dataset["date"].iloc[:(init_time+1+t*period)],
                y=df_dataset["Average USD (7-day)"].iloc[:(init_time+1+t*period)],
                name="Price USD",
                line=dict(color='#2ca02c'),
                visible=False,
                showlegend=False),
                )
            m = Prophet()
            m.fit(df_prophet.iloc[:(init_time+t*period)])
            date_future = m.make_future_dataframe(periods=period)
            df_forecast = m.predict(date_future)
            df_forecast = df_forecast.iloc[-period:]

            corr_factor = 0
            try :
                if df_forecast['yhat'].iloc[0] < df_dataset["Average USD (7-day)"].iloc[init_time+t*period]:
                    corr_factor = np.abs(df_dataset["Average USD (7-day)"].iloc[init_time+t*period] - df_forecast['yhat'].iloc[0])
                else:
                    corr_factor = - np.abs(df_dataset["Average USD (7-day)"].iloc[init_time+t*period] - df_forecast['yhat'].iloc[0])
            except:
                pass

            # Add lower bound
            fig.add_trace(go.Scatter(
                x=df_forecast['ds'],
                y=df_forecast['yhat_lower']+corr_factor,
                mode='lines',
                line=dict(width=0),
                hoverinfo='skip',
                visible=False,
                showlegend=False),
                )

            fig.add_trace(go.Scatter(
                name='Forecast prediction',
                x=df_forecast['ds'],
                y=df_forecast['yhat']+corr_factor,
                mode='lines',
                line=dict(color='firebrick'),
                fillcolor='rgba(255, 255, 255, 0.2)',
                fill='tonexty',
                visible=False,
                showlegend=False),
                )

            # Add upper bound
            fig.add_trace(go.Scatter(
                x=df_forecast['ds'],
                y=df_forecast['yhat_upper']+corr_factor,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 255, 255, 0.2)',
                fill='tonexty',
                hoverinfo='skip',
                visible=False,
                showlegend=False),
                )

        # Create and add slider
        steps = []
        for i in range((len(fig.data)//4)-1):
            step = dict(
                method="restyle",
                label="",
                args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            )
            if i != 0:
                step["args"][0]["visible"][4*i+1] = True  # Real
                step["args"][0]["visible"][4*i+2] = True  # Forecast lower bound
                step["args"][0]["visible"][4*i+3] = True  # Forecast
                step["args"][0]["visible"][4*i+4] = True  # Forecast upper bound

                for i in range(i,0,-1):
                    step["args"][0]["visible"][4*i] = True  # Previous Forecast upper bound
                    step["args"][0]["visible"][4*i-1] = True  # Previous Forecast 
                    step["args"][0]["visible"][4*i-2] = True  # Previous Forecast lower bound

            step["args"][0]["visible"][0] = True  # Dummy for axes 
            step["args"][0]["visible"][1] = True  # Real
            step["args"][0]["visible"][2] = True  # Forecast lower bound
            step["args"][0]["visible"][3] = True  # Forecast
            step["args"][0]["visible"][4] = True  # Forecast upper bound
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"visible": False},
            pad={"t": 1},
            steps=steps[:-6],
            yanchor="bottom"
        )]

        fig.update_layout(sliders=sliders)

        # Update layout properties
        fig.update_layout(title_text="Denoise Nft :" + collection_name, hovermode="x")
        fig.update_layout(
            yaxis=dict(
                title="NFT Price USD"
                )
        )
        fig.update_layout(template="plotly_dark")
        #fig.show()
        #exit()
        py.plot(fig, filename = "Denoise Nft :" + collection_name, auto_open=True)