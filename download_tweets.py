import tweepy
import json
import sys


CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_TOKEN = ''
ACCESS_SECRET = ''

TOTAL_DOWNLOAD = 100000
BUFFER = 1000

TWEETER_FEEDS = [
	'nytimes', 'thesun', 'thetimes', 'ap', 'cnn',
	'bbcnews', 'cnet', 'msnuk', 'telegraph']
#				  usatoday, wsj, washingtonpost, bostonglobe, newscomauhq, skynews, sfgate,##
#				  ajenglish, independent, guardian, latimes, reutersagency, abc, business, bw, time,]

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)
# public_tweets = api.home_timeline()

for feed in TWEETER_FEEDS:
	user = api.get_user(feed)
	print(user.screen_name)
	# print(user.followers_count)

	tweet_data_file = open('downloaded_tweets/'+feed+'-tweet.json', 'w')

	first_tweet = [status._json for status in tweepy.Cursor(api.user_timeline, id=feed).items(1)]
	#print first_tweet[0]['text']
	tweet_data_file.writelines(json.dumps(first_tweet[0], encoding="utf-8") + '\n')

	max_id = first_tweet[0]['id'] - 1
	num_tweets_download = 1
	tweet_objects = []

	while num_tweets_download <= TOTAL_DOWNLOAD+1:
		tweet_objects = [status._json for status in tweepy.Cursor(api.user_timeline, id=feed, max_id=max_id).items(BUFFER)]
		print len(tweet_objects)

		try:
			max_id = tweet_objects[-1]['id'] - 1
		except:
			e = sys.exc_info()[0]
			print "<p>Error: %s</p>" % e
			break

		for tweet in tweet_objects:
			tweet_data_file.writelines(json.dumps(tweet, encoding="utf-8")+'\n')
		num_tweets_download = num_tweets_download + len(tweet_objects)
		print "Num Tweets Download: ", num_tweets_download
		del tweet_objects
		tweet_objects = []


	tweet_data_file.close()
