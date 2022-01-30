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
import utils as ut
import time
from pytrends.request import TrendReq
from dateutil import parser
twitter_keys = "/home/francoismasson/denoise_nft/twitter_key.json"


def tweppy_token():
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
        member = ut.scrap_url(url, key="meta", property="og:description")
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
