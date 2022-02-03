#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Use api such as twitter to gather info regarding collections

Documentation for tweepy can be found here
https://docs.tweepy.org/en/stable/api.html
https://www.geeksforgeeks.org/python-api-followers-in-tweepy/
"""
import tweepy as tw
import json
import denoise.misc.sql as sql
import denoise.misc.utils as ut
import time
from pytrends.request import TrendReq
from dateutil import parser
from opensea import CollectionStats, Collection
from pathlib import Path


def tweppy_token():
    twitter_keys = ut.get_twitter_keys()
    with open(twitter_keys) as json_file:
        keys = json.load(json_file)
    return keys


def tweepy_api(keys):
    auth = tw.AppAuthHandler(keys["api_key"], keys["api_key_secret"])
    return tw.API(auth, wait_on_rate_limit=True)


def get_twitter_metrics(df, url, date, index):
    keys = tweppy_token()
    api = tweepy_api(keys)
    screen_name = url.split("/")[-1]
    try:
        user = api.get_user(screen_name=screen_name)
        # fetching the followers_count
        followers_count = user.followers_count
        df.loc[index, "twitter_followers"] = followers_count
        tweets = \
            api.user_timeline(screen_name=screen_name,
                              # 200 is the maximum allowed count
                              count=200)
        post = 0
        retweet = 0
        like = 0
        date = parser.parse(date).date()
        for tweet in tweets:
            if tweet.created_at.date() >= date:
                post += 1
                retweet += tweet.retweet_count
                like += tweet.favorite_count
        if post > 0:
            retweet /= post
            like /= post
        df.loc[index, "twitter_post"] = post
        df.loc[index, "twitter_retweet"] = retweet
        df.loc[index, "twitter_like"] = like
    except Exception as e:
        print(screen_name)
        print(e)


def get_discord_metrics(df, url, index):
    count = None
    try:
        member = ut.scrap_url(url,
                              key="meta",
                              property="og:description")
        time.sleep(1)
        if member:
            # using List comprehension + isdigit() +split()
            # getting numbers from string
            count = member.replace(",", "")
            count = [int(i) for i in count.split() if i.isdigit()]
            if count:
                count = max(count)
            else:
                count = None
    except Exception as e:
        print(e)
    df.loc[index, "discord_members"] = count


def get_google_trends(df, name, date, index):
    # https://hackernoon.com/how-to-use-google-trends-api-with-python
    trends = TrendReq(hl='en-US', tz=360)
    # list of keywords to get data
    kw_list = name.lower()
    kw_list = "machine"
    # Google delay of 3 days with granularity of 1 month
    trends.build_payload([kw_list], cat=0, timeframe='today 1-m')
    data = trends.interest_over_time()
    score = 0
    if len(data) > 0:
        score = data[kw_list].mean()
    time.sleep(1)
    df.loc[index, "google_trend"] = score


def get_opensea_metrics(df, name, index, key="stats"):
    stats = CollectionStats(collection_slug=name).fetch()
    if key in list(stats.keys()):
        df.loc[index, list(stats[key].keys())] = list(stats[key].values())


def get_opensea_infos(name):
    infos = Collection(collection_slug=name).fetch()
    return infos


def get_opensea_filters():
    volume = ["one_day_volume",
              "seven_day_volume",
              "thirty_day_volume",
              "total_volume"
              ]
    # None = all categories
    categories = [None,
                  "new",
                  "art",
                  "collectibles",
                  "domain-names",
                  "music",
                  "photography-category",
                  "sports",
                  "trading-cards",
                  "utility",
                  "virtual-worlds"
                  ]
    # None = all chains
    chains = [None,
              "ethereum",
              "matic",
              "klaytn"
              ]
    return [volume, categories, chains]


def get_social_column():
    columns = ["google_trend",
               "twitter_followers",
               "twitter_post",
               "twitter_retweet",
               "twitter_like",
               "discord_members"]
    return columns


def get_market_column():
    columns = ['one_day_volume',
               'one_day_change',
               'one_day_sales',
               'one_day_average_price',
               'seven_day_volume',
               'seven_day_change',
               'seven_day_sales',
               'seven_day_average_price',
               'thirty_day_volume',
               'thirty_day_change',
               'thirty_day_sales',
               'thirty_day_average_price',
               'total_volume',
               'total_sales',
               'total_supply',
               'count',
               'num_owners',
               'average_price',
               'num_reports',
               'market_cap',
               'floor_price']
    return columns


def add_column(df, name, columns):
    save = False
    for col in columns:
        if col not in df.columns:
            save = True
            df[col] = 0
    if save:
        sql.to_sql(df, sql_path=name)
    return df
