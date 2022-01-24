# import the module
import tweepy as tw
import json

if __name__ == "__main__":

    # Documentation for tweepy can be found here
    # https://docs.tweepy.org/en/stable/api.html

    # https://www.geeksforgeeks.org/python-api-followers-in-tweepy/

    twitter_keys = "../twitter_key.json"
    with open(twitter_keys) as json_file:
        keys = json.load(json_file)
    
    auth = tw.AppAuthHandler(keys["api_key"], keys["api_key_secret"])
    api = tw.API(auth,wait_on_rate_limit=True)
    # authorization of consumer key and consumer secret
    
    # the ID of the user is an option 
    id = 57741058
    # the screen_name is another one
    screen_name = "scoinaldo"
    # fetching the user
    user = api.get_user(screen_name=screen_name)
  
    # fetching the followers_count
    followers_count = user.followers_count
  
    print("The number of followers of the user are : " + str(followers_count))

    # Already have a script get tweet between specific date. I can also check number of retweet and like
    # the ID of the status
    id = 1272771459249844224
    
    # fetching the status
    status = api.get_status(id)
    
    # fetching the retweet_count attribute
    retweet_count = status.retweet_count 
    
    print("The number of time the status has been retweeted is : " + str(retweet_count))