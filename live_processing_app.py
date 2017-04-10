import tweepy
import json
import sys


CONSUMER_KEY = 'kCTatb4tsfq5lihQAmHhiKkjO'
CONSUMER_SECRET = 'K5gx7jzQj7KcgK4sYLdgCANM4ywk7jOhmMGe3kvTX5HYm1W2Ed'
ACCESS_TOKEN = '158395360-BhUmfsA3swXtZAaxAKk8XNqEbXsdek4JRvgvZHDh'
ACCESS_SECRET = 'vDUE1cCL1n7BLrRKYlGvwQHeKwbF1YX6IrxpkjTs6XQ4l'

TOTAL_DOWNLOAD = 100000
BUFFER = 1000

TWEETER_FEEDS = [
	#'nytimes', 'thesun', 'thetimes', 'ap', 'cnn',
	'bbcnews', 'cnet', 'msnuk', 'telegraph']
#				  usatoday, wsj, washingtonpost, bostonglobe, newscomauhq, skynews, sfgate,##
#				  ajenglish, independent, guardian, latimes, reutersagency, abc, business, bw, time,]

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)