import tweepy
import json
import sys
from sklearn.cluster import KMeans
import numpy as np
import random
import re
import gensim
import nltk
from keras.preprocessing import sequence
import pickle
from collections import defaultdict
import pandas as pd
import math
import time

import warnings

warnings.filterwarnings("ignore")

STOPWORDS = nltk.corpus.stopwords.words('english')

emoticons_str = r"""
		(?:
			[:=;] # Eyes
			[oO\-]? # Nose (optional)
			[D\)\]\(\]/\\OpP] # Mouth
		)"""

regex_str = [
	emoticons_str,
	r'<[^>]+>',  # HTML tags
	r'(?:@[\w_]+)',  # @-mentions
	r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
	r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
	r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
	r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
	r'(?:[\w_]+)'  # other words
	# r'(?:\S)'  # anything else
]

number_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
number_re = re.compile(r'^' + number_str + '$', re.VERBOSE | re.IGNORECASE)

CONSUMER_KEY = 'kCTatb4tsfq5lihQAmHhiKkjO'
CONSUMER_SECRET = 'K5gx7jzQj7KcgK4sYLdgCANM4ywk7jOhmMGe3kvTX5HYm1W2Ed'
ACCESS_TOKEN = '158395360-BhUmfsA3swXtZAaxAKk8XNqEbXsdek4JRvgvZHDh'
ACCESS_SECRET = 'vDUE1cCL1n7BLrRKYlGvwQHeKwbF1YX6IrxpkjTs6XQ4l'

BUFFER = 500
NUMBER_CLUSTERS = 3
RANK_THRESHOLD = {}
RANK_THRESHOLD[0] = 10000
RANK_THRESHOLD[1] = 1000
RANK_THRESHOLD[2] = 1000

TWEETER_FEEDS = [
	'nytimes', 'thesun', 'thetimes', 'ap', 'cnn',
	'bbcnews', 'cnet', 'msnuk', 'telegraph']


def tokenize(s):
	return tokens_re.findall(s)


def preprocess(s):
	tokens = tokenize(s)
	# tokens = [token.lower() if emoticon_re.search(token) == False and token not in STOPWORDS else  for token in tokens]
	tokens = map(lambda token: token.lower(),
				 filter(lambda token: emoticon_re.search(token) is None
									  and token not in STOPWORDS
									  and token.find('http') == -1
									  and number_re.search(token) is None
						, tokens))

	return tokens


def load_google_word2vec_model(path_google_word2vec):
	google_model = gensim.models.KeyedVectors.load_word2vec_format(path_google_word2vec,
																   binary=True)
	return google_model


def process_features(tweet_objects, google_model):
	max_review_length = 7500
	tweet_vectors = []
	for data_dict in tweet_objects:
		sentence = preprocess(data_dict['text'])
		sentence_vector = []
		for word in sentence:
			try:
				tmp_vec = google_model.word_vec(word).tolist()  # np.array([1,1,1,1]).tolist()
			except:
				e = sys.exc_info()[0]
				# print "<p>Error: %s</p>" % e
				tmp_vec = []

			sentence_vector += tmp_vec

		dummy_list = []
		dummy_list.append(sentence_vector)
		sentence_vector = dummy_list
		# print sentence_vector
		sentence_vector = sequence.pad_sequences(sentence_vector,
												 maxlen=max_review_length,
												 padding='post', truncating='post',
												 dtype='float32')
		tweet_vectors.append(sentence_vector[0])

	return tweet_vectors


def _get_clusters(X, model):
	clus = model.predict(X)
	return clus


def _get_features_for_score(tweet_objects):
	fields_dict = defaultdict(list)
	for data_dict in tweet_objects:
		# print '\n'.join(np.unique(data_dict.keys()))
		fields_dict['favorite_count'].append(
			int(data_dict['favorite_count']) if data_dict.get('favorite_count') is not None else 0)
		fields_dict['favorited'].append(
			bool(data_dict['favorited']) if data_dict.get('favorited') is not None else False)
		fields_dict['possibly_sensitive'].append(
			bool(data_dict['favorited']) if data_dict.get('possibly_sensitive') is not None else False)
		fields_dict['retweet_count'].append(int(data_dict['retweet_count']))
		fields_dict['retweet_count_ln'].append(math.log(int(data_dict['retweet_count']) + 1))
		fields_dict['retweeted'].append(bool(data_dict['retweeted']))
		fields_dict['retweeted_status'].append(1 if data_dict.get('retweeted_status') is not None else 0)
		entities_encode = [0, 0, 0]
		if len(data_dict['entities']['hashtags']) > 0:
			entities_encode[0] = 1
		if len(data_dict['entities']['urls']) > 0:
			entities_encode[1] = 1
		if len(data_dict['entities']['user_mentions']) > 0:
			entities_encode[2] = 1
		fields_dict['entities'].append(entities_encode)

	X = pd.DataFrame()
	X['favorited'] = fields_dict['favorited']
	X['retweeted_status'] = fields_dict['retweeted_status']
	X['retweeted'] = fields_dict['retweeted']
	X['entities_h'] = map(lambda x: x[0], fields_dict['entities'])
	X['entities_u'] = map(lambda x: x[0], fields_dict['entities'])
	X['entities_m'] = map(lambda x: x[0], fields_dict['entities'])
	X['favorite_count'] = fields_dict['favorite_count']
	X['possibly_sensitive'] = fields_dict['possibly_sensitive']
	y = pd.DataFrame()
	y['target'] = fields_dict['retweet_count_ln']
	y_true = pd.DataFrame()
	y_true['count'] = fields_dict['retweet_count']
	return X, y


def _get_tweet_scores(tweet_objects, model):
	X, y = _get_features_for_score(tweet_objects)
	pred = model.predict(X)
	return pred


class rank_object:
	def __init__(self):
		self.posted = True
		self.score = -1
		self.text = ''
		self.id = -1
		self.created_at = ''
		self.url = ''
		self.retweet_count = -1
		self.weighted_score = -1
		self.retweet_id = None


def find_top_tweets(clusters, scores, tweet_objects):
	rank_dict = {}
	for i in range(NUMBER_CLUSTERS):
		rank_dict[i] = rank_object()

	for i, clus in enumerate(clusters):
		if rank_dict[clus].weighted_score < (scores[i] * tweet_objects[i]['retweet_count']):
			rank_dict[clus].score = scores[i]
			rank_dict[clus].weighted_score = (scores[i] * tweet_objects[i]['retweet_count'])
			rank_dict[clus].index = i
			rank_dict[clus].text = tweet_objects[i]['text']
			rank_dict[clus].id = tweet_objects[i]['id']
			rank_dict[clus].posted = False
			rank_dict[clus].created_at = tweet_objects[i]['created_at']
			rank_dict[clus].retweet_count = tweet_objects[i]['retweet_count']
			rank_dict[clus].retweet_id = tweet_objects[i]['retweeted_status']['id'] if tweet_objects[i].get(
				'retweeted_status') is not None else None
	return rank_dict


def write_tweets(top_tweets, post_file):
	for i in range(NUMBER_CLUSTERS):
		# print i, len(top_tweets[i])
		for tweet in top_tweets[i]:
			if tweet.posted == False and tweet.weighted_score >= RANK_THRESHOLD[i]:
				line = str(i) + ',' + str(tweet.id) + ',' + str(tweet.created_at) + ',' + \
					   str(tweet.weighted_score) + ',' + str(tweet.score) + ',' + str(tweet.retweet_count) + ',' \
					   + tweet.text.encode('utf8') + '\n'
				post_file[i].write(line)
				tweet.posted = True
		post_file[i].flush()


def process_tweets(path_google_word2vec):
	# get tweeter access
	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

	# load google word2vec
	google_model = load_google_word2vec_model(path_google_word2vec)

	# load clustering model
	kmeans = pickle.load(open('kmeans_model.sav', 'rb'))

	# load ranking model
	rf = pickle.load(open('rf_model.sav', 'rb'))

	# all top tweets from each download
	total_top_tweets = defaultdict(list)

	# posting tweet file
	tweet_file = []
	for i in range(NUMBER_CLUSTERS):
		tweet_file.append(open('posted_tweets_clus_' + str(i + 1) + '.csv', 'w'))
		line = 'cluster_id, tweet_id, created_at, weighted_score, score, retweet_count, text\n'
		tweet_file[i].write(line)

	# tweet dump file
	tweet_dump = open('tweet_dump.json', 'w')

	api = tweepy.API(auth)

	num_tweets_download = 0
	more_tweets = True
	last_id = {}
	# for feed in TWEETER_FEEDS:
	# 	last_id[feed] = 0 #850000000000000000

	last_id['nytimes'] = 853319033424314368  # 852854216603435008
	last_id['thesun'] = 853320020012806144  # 852945509279047682
	last_id['thetimes'] = 853262389231456257  # 852421095965773824
	last_id['ap'] = 853320221683306496  # 852640846763655170
	last_id['cnn'] = 853319517975576576  # 852780968947490816
	last_id['bbcnews'] = 853315775528030208  # 852306647867420672
	last_id['cnet'] = 853318276713197570  # 852650669161017344
	last_id['msnuk'] = 852972496064311297  # 850373113925783557
	last_id['telegraph'] = 853317000726274048  # 852551974021079040

	while True:
		total_tweet_objects = []
		more_tweets = True
		for feed in TWEETER_FEEDS:
			more_tweets = True
			user = api.get_user(feed)
			recent_id = last_id[feed]
			print(user.screen_name)

			try:
				tweet_objects = [status._json for status in
								 tweepy.Cursor(api.user_timeline, id=feed).items(BUFFER)]
			except:
				e = sys.exc_info()[0]
				print "<p>Error: %s</p>" % e
				tweet_objects = []

			if len(tweet_objects) > 0:
				recent_id = int(tweet_objects[0]['id'])
			else:
				more_tweets = False

			while more_tweets:
				# print len(tweet_objects)
				# print tweet_objects[0]['id']
				for tweet in tweet_objects:
					if last_id[feed] >= int(tweet['id']):
						more_tweets = False
						break
					total_tweet_objects.append(tweet)

				# cursor only gives unread tweets only
				try:
					tweet_objects = [status._json for status in
									 tweepy.Cursor(api.user_timeline, id=feed).items(BUFFER)]
				except:
					e = sys.exc_info()[0]
					# print "<p>Error: %s</p>" % e
					tweet_objects = []
					more_tweets = False

			last_id[feed] = recent_id
			# print feed, last_id[feed]
			print "new len total_tweets: ", len(total_tweet_objects)

		if len(total_tweet_objects) != 0:
			print "final len total tweets: ", len(total_tweet_objects)

			tweet_vectors = process_features(total_tweet_objects, google_model)
			clusters = _get_clusters(list(tweet_vectors), kmeans)
			predicted_scores = _get_tweet_scores(total_tweet_objects, rf)
			top_tweets = find_top_tweets(clusters, predicted_scores, total_tweet_objects)

			for i in range(NUMBER_CLUSTERS):
				tweet_added = False
				for j, item in enumerate(total_top_tweets[i]):
					if tweet_added == False and (
									item.id == top_tweets[i].retweet_id or item.retweet_id == top_tweets[i].retweet_id):
						top_tweets[i].posted = total_top_tweets[i][j].posted
						total_top_tweets[i][j] = top_tweets[i]
						tweet_added = True
						break
				if tweet_added == False:
					total_top_tweets[i].append(top_tweets[i])

			write_tweets(total_top_tweets, tweet_file)

			# writing all tweets to dump file
			for index, tweet in enumerate(total_tweet_objects):
				tweet['predicted_score'] = predicted_scores[index]
				tweet['weight_score'] = tweet['retweet_count'] * predicted_scores[index]
				tweet['cluster'] = clusters[index] + 1
				tweet_dump.writelines(json.dumps(tweet, encoding="utf-8") + '\n')
			tweet_dump.flush()

		print "going to sleep .... %s" % time.ctime()
		time.sleep(600)
		print "wake up .... %s" % time.ctime()


if __name__ == '__main__':
	if len(sys.argv) >= 2:
		process_tweets(sys.argv[1])
	else:
		print "Error: Google word2vec path not specified"
		print "Please download word2vec from here https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit"
		print "Give the path to extraced GoogleNews-vectors-negative300.bin"
