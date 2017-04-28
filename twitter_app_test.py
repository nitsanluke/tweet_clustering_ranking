from twython import Twython
import time

#########
import tweepy
import json


CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_TOKEN = ''
ACCESS_SECRET = ''

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print(tweet.text)

user = api.get_user('nytimes')

print(user.screen_name)
print(user.followers_count)

print user.

# for friend in user.friends():
#   print(friend.screen_name)

#for status in tweepy.Cursor(api.user_timeline, id='nytimes').items():
	# process status here
	#print status
	#break
