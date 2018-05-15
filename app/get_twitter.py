
# This file is used to extract data from Twitter.

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler , Stream , API
import pandas as pd
import numpy as np


# Authentication Details
consumer_key = " "
consumer_secret = " "
access_token = " "
access_token_secret = " "

# Creating the authentication object
auth = OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = API(auth) 


class StdOutListener(StreamListener):
    def __init__(self,api,count):
        self.api = api
        self.counter = count + 1
        
    def on_status(self,data):
        self.counter -= 1
        if self.counter > 0 :
            arr.append([data.created_at ,data.text.encode('utf-8')])
        else :
            return False
  

  arr = []

query = "#padman"
streaming = StdOutListener(api,3)
stream = Stream(auth,streaming)
track = query.split(' OR ')
stream.filter(track = track)

# loading the array of tweets in a dataframe
df_write = pd.DataFrame(arr)
df_write.columns = ['Date','Tweet']

# saving the dataframe as a csv file
df_write.to_csv('Tweets_test.csv' , header = True , index = False)
    
# loading the saved twitter csv file
df_read = pd.read_csv('Tweets_test.csv')
    
    
    
    
    
    
    
    
    
    
    
    