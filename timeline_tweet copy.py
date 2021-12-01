import datetime
from inspect import trace
import tensorflow as tf
import tweepy as tw
import json
import pandas as pd
import os
import joblib
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import chart_studio
import chart_studio.plotly as py

username = 'francois.masson' # your username
api_key = '6jAq2ymELJmOJWBQ9Tme' # your api key - go to profile > settings > regenerate key in plotlyaccount
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

def model_tf(checkpoint_path):
    # Build the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.load_weights(checkpoint_path)

    return model

    
if __name__ == "__main__":

    # Documentation for tweepy can be found here
    # https://docs.tweepy.org/en/stable/api.html

    twitter_keys = "twitter_key.json"
    with open(twitter_keys) as json_file:
        keys = json.load(json_file)
    
    auth = tw.AppAuthHandler(keys["api_key"], keys["api_key_secret"])
    api = tw.API(auth,wait_on_rate_limit=True)

    model = model_tf(checkpoint_path = "training_1/cp.ckpt")
    # with new data
    scaler = joblib.load('scaler.joblib')

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

        # Predictionl
        x_test_df = df_dataset.iloc[: , 1:]
        index = df_dataset.columns.get_loc(target)
        x_test = scaler.transform(x_test_df)
        x_test = pd.DataFrame(x_test, columns=x_test_df.columns.values)
        x_test = x_test.drop(target, axis=1).values

        # inference
        multiplied_by = scaler.scale_[index]
        added = scaler.min_[index]
        prediction = model.predict(x_test)
        prediction -= added
        prediction /= multiplied_by

        df_dataset["prediction"] = prediction

        # Create figure with secondary y-axis
        #fig = make_subplots(rows=2,cols=1,specs=[[{"secondary_y": True}],
        #                                         [{"secondary_y": True}]])

        fig = go.Figure()

        # Also create a dummy invisible graph to fix length
        fig.add_trace(go.Scatter(
                x=df_collections_name["time"],
                y=df_collections_name[param],
                name="Twitter " + param,
                line=dict(color='rgba(0,0,0,0)'),
                visible=True),
                #row=1, col=1, secondary_y=False
                )
        
        # Not period but daterange !!! 
        period = 3
        traces = 4
        for i in range(1,period+1):
            fig.add_trace(go.Scatter(
                x=df_collections_name["time"].iloc[:(len(df_collections_name["time"])*i//period)],
                y=df_collections_name[param].iloc[:(len(df_collections_name["time"])*i//period)],
                name="Twitter " + param,
                line=dict(color='#1f77b4'),
                visible=False),
                #row=1, col=1, secondary_y=False
                )

            fig.add_trace(go.Scatter(
                x=df_dataset["date"].iloc[:(len(df_dataset["date"])*i//period)],
                y=df_dataset[collection_name.lower()].iloc[:(len(df_dataset["date"])*i//period)],
                name="google_queries",
                yaxis="y2",
                line=dict(color='#ff7f0e'),
                visible=False),
                #row=1,col=1,secondary_y=True
                )

            fig.add_trace(go.Scatter(
                x=df_dataset["date"].iloc[:(len(df_dataset["date"])*i//period)],
                y=df_dataset["Average USD (7-day)"].iloc[:(len(df_dataset["date"])*i//period)],
                name="Price USD",
                yaxis="y3",
                line=dict(color='#2ca02c'),
                visible=False),
                #row=2,col=1, secondary_y=False
                )

            y_upper = 1.1*df_dataset["prediction"]
            y_lower = 0.90*df_dataset["prediction"]

            fig.add_trace(go.Scatter(
                x=df_dataset["date"].tolist() + df_dataset["date"].tolist()[::-1], # x, then x reversed
                y=y_upper.tolist() + y_lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,255,255,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="Prediction Price USD",
                yaxis="y4")
                )

        for i in range(1,traces+1):
            # Make nth trace visible
            fig.data[len(fig.data)-i].visible = True

        # Create and add slider
        steps = []
        for i in range((len(fig.data)-1)//(traces)):
            step = dict(
                method="restyle",
                label="",
                args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            )
            for j in range(traces):
                k = len(fig.data)//(period)
                step["args"][0]["visible"][i*k+j+1] = True  # Toggle i'th trace to "visible"
            step["args"][0]["visible"][0] = True  # Toggle i'th trace to "visible"  
            steps.append(step)

        sliders = [dict(
            active=period-1,
            currentvalue={"visible": False},
            pad={"t": 1},
            steps=steps,
            yanchor="bottom"
        )]

        fig.update_layout(sliders=sliders)

        # fig.add_trace(go.Scatter(
        #     x=df_dataset["date"],
        #     y=df_dataset["prediction"],
        #     name="Prediction Price USD",
        #     yaxis="y4"),
        #     #row=2,col=1, secondary_y=True
        #     )

        # Update layout properties
        fig.update_layout(title_text="Denoise Nft :" + collection_name, hovermode="x")
        # fig.update_layout(
        #     xaxis=dict(
        #         rangeslider=dict(
        #             visible=True
        #             ),
        #         type="date"
        #         )
        #     )
        fig.update_layout(
            yaxis=dict(
                title="Social trends"
                ),
            yaxis2=dict(
                title="google_queries",
                anchor="free",
                overlaying="y",
                side="left",
                position=0.05,
                visible=False,
                ),
            yaxis3=dict(
                title="NFT Price USD",
                anchor="x",
                overlaying="y",
                side="right"
                ),
            yaxis4=dict(
                title="Prediction Price USD",
                anchor="free",
                overlaying="y",
                side="right",
                position=0.95,
                visible = False,
                )
            )
        fig.update_layout(
            updatemenus=[
                dict(
                    font=dict({"color":"white"}),
                    bgcolor= "black",
                    bordercolor = "white",
                    type = "dropdown",
                    #type="buttons",
                    direction="right",
                    buttons=list([
                        dict(label="Social",
                            method="update",
                            args=[{"visible": [True, True, False, False]},
                                 {"style":"dark"}]),
                        dict(label="NFT Market",
                            method="update",
                            args=[{"visible": [False, False, True, True]},
                                 {"style":"dark"}]),
                        dict(label="Google",
                            method="update",
                            args=[{"visible": [True, False, False, False]},
                                 {"style":"dark"}]),
                        dict(label="Twitter",
                            method="update",
                            args=[{"visible": [False, True, False, False]},
                                 {"style":"dark"}]),
                        dict(label="Price Prediction",
                            method="update",
                            args=[{"visible": [False, False, False, True]},
                                 {"style":"dark"}]),
                        dict(label="Real Price",
                            method="update",
                            args=[{"visible": [False, False, True, False]},
                                 {"style":"dark"}]),
                        dict(label="All",
                            method="update",
                            args=[{"visible": [True, True, True, True]},
                                 {"style":"dark"}]),
                    ]),
                )
            ])
        fig.update_layout(template="plotly_dark")
        fig.show()
        exit()
        #py.plot(fig, filename = "Denoise Nft :" + collection_name, auto_open=True)