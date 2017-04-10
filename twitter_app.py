from twython import Twython
import time


CONSUMER_KEY = 'kCTatb4tsfq5lihQAmHhiKkjO'
CONSUMER_SECRET = 'K5gx7jzQj7KcgK4sYLdgCANM4ywk7jOhmMGe3kvTX5HYm1W2Ed'
ACCESS_KEY = '158395360-BhUmfsA3swXtZAaxAKk8XNqEbXsdek4JRvgvZHDh'
ACCESS_SECRET = 'vDUE1cCL1n7BLrRKYlGvwQHeKwbF1YX6IrxpkjTs6XQ4l'

twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_KEY, ACCESS_SECRET)
# lis = [467020906049835008] ## this is the latest starting tweet id
lis = []
for i in range(0, 1):

	print "te"
	user_timeline = twitter.get_user_timeline(screen_name="nytimes", count=200, include_retweets=False)
	#time.sleep(300) ## 5 minute rest between api calls

	for tweet in user_timeline:
		print tweet['text'] ## print the tweet
		lis.append(tweet['id']) ## append tweet id's

#########
import tweepy
import json


CONSUMER_KEY = 'kCTatb4tsfq5lihQAmHhiKkjO'
CONSUMER_SECRET = 'K5gx7jzQj7KcgK4sYLdgCANM4ywk7jOhmMGe3kvTX5HYm1W2Ed'
ACCESS_TOKEN = '158395360-BhUmfsA3swXtZAaxAKk8XNqEbXsdek4JRvgvZHDh'
ACCESS_SECRET = 'vDUE1cCL1n7BLrRKYlGvwQHeKwbF1YX6IrxpkjTs6XQ4l'

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
