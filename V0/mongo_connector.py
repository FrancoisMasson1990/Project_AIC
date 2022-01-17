import quandl
import pandas as pd
import pymongo
from pymongo import MongoClient

password = "121Dindon!"
database = "Denoise"

# Get dataset example
data = quandl.get("BSE/SENSEX", authtoken="",start_date = "2019-01-01")

# Connect to MongoDB
client = MongoClient(f"mongodb+srv://francoismasson:{password}@clusterdenoise.o6lva.mongodb.net/{database}?retryWrites=true&w=majority")
db = client[database]
collection = db['Test']
data.reset_index(inplace=True)
data_dict = data.to_dict("records")
# Insert collection
collection.insert_many(data_dict)
