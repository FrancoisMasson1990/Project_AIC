import os
import pandas as pd
import numpy as np
import joblib
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import plotly.graph_objects as go

import chart_studio
import chart_studio.plotly as py

username = 'francois.masson' # your username
api_key = '6jAq2ymELJmOJWBQ9Tme' # your api key - go to profile > settings > regenerate key in plotlyaccount
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

df = []
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
        "Doodles"]

test_data = "CryptoPunks"
#test_data = "Decentraland"
for dir in os.listdir(dataset):
    #if dir != test_data and dir in nfts:
    if dir in nfts:
        asset = dir.lower()
        if os.path.exists(os.path.join(dataset,dir,"dataset.csv")):
            data = pd.read_csv(os.path.join(dataset,dir,"dataset.csv"))
            data = data.rename(columns={asset: "google_queries"})
            df.append(data)

asset = test_data.lower()
df_test = pd.read_csv(os.path.join(dataset,test_data,"dataset.csv"))
df_test = df_test.rename(columns={asset: "google_queries"})
df_test = df_test.iloc[: , 1:]

#read in training data
df = pd.concat(df)
df = df.iloc[: , 1:]

target = 'Average USD (7-day)'
index = df.columns.get_loc(target)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(df)
scaled_test = scaler.transform(df_test)

multiplied_by = scaler.scale_[index]
added = scaler.min_[index]

joblib.dump(scaler, 'scaler.joblib') 

scaled_train_df = pd.DataFrame(scaled_train, columns=df.columns.values)
scaled_test_df = pd.DataFrame(scaled_test, columns=df_test.columns.values)

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

metric ='val_accuracy'
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model
x_train = scaled_train_df.drop(target, axis=1).values
y_train = scaled_train_df[[target]].values
#x_train, x_val, y_train, y_val = \
#    train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Train the model
x_test = scaled_test_df.drop(target, axis=1).values
y_test = scaled_test_df[[target]].values

if os.path.exists(checkpoint_dir):
    # Loads the weights 
    model.load_weights(checkpoint_path)
else:
    # Train the model
    model.fit(x_train,
              y_train,
              #validation_data=(x_val, y_val),
              epochs=50,
              shuffle=True,
              verbose=2,
              callbacks=[cp_callback])

# inference
prediction = model.predict(x_test)
prediction -= added
prediction /= multiplied_by

filter_date = datetime.datetime(2021,1,1)

path = os.path.join(dataset,test_data,"dataset.csv")
price_usd = pd.read_csv(path)
price_usd[price_usd.columns[0]] = pd.to_datetime(price_usd[price_usd.columns[0]])
price_usd = price_usd[price_usd[price_usd.columns[0]]>filter_date]
price_usd["prediction"] = prediction

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=price_usd[price_usd.columns[0]],
    y=price_usd[price_usd.columns[3]],
    name="Price USD"
))

fig.add_trace(go.Scatter(
    x=price_usd[price_usd.columns[0]],
    y=price_usd["prediction"],
    name="Price USD Prediction",
    yaxis="y2"
))

# Create axis objects
fig.update_layout(
    xaxis=dict(
        domain=[0.2, 0.7]
    ),
    yaxis=dict(
        title="Price USD"),
    yaxis2=dict(
        title="Price USD Prediction",
        anchor="x",
        overlaying="y",
        side="right"
    ),
)

# Update layout properties
fig.update_layout(title_text=asset)
fig.show()
#py.plot(fig, filename = asset, auto_open=True)