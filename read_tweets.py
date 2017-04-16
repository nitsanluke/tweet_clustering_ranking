import json
import sys
import re
import gensim
import nltk
import numpy as np
from keras.preprocessing import sequence

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
		#r'(?:\S)'  # anything else
		]

number_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
number_re = re.compile(r'^' + number_str + '$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
	return tokens_re.findall(s)

def preprocess(s):
	tokens = tokenize(s)
	#tokens = [token.lower() if emoticon_re.search(token) == False and token not in STOPWORDS else  for token in tokens]
	tokens = map(lambda token:token.lower(),
				 filter(lambda token: emoticon_re.search(token) is None
									  and token not in STOPWORDS
									  and token.find('http') == -1
									  and number_re.search(token) is None
						, tokens))

	return tokens


def load_google_word2vec_model():
	google_model = gensim.models.KeyedVectors.load_word2vec_format('google-word2vec/GoogleNews-vectors-negative300.bin',
																   binary=True)
	return google_model


def main():
	# print "argv 1 - tweet data file"
	# print "argv 2 - feature file to save"

	TWEETER_FEEDS = [
		 'nytimes', 'thesun', 'thetimes', 'ap', 'cnn',
		'bbcnews', 'cnet', 'msnuk', 'telegraph'
		]

	google_model = load_google_word2vec_model()
	max_review_length = 7500

	for feed in TWEETER_FEEDS:
		print feed
		data_File_NYTimes = open('downloaded_tweets/'+feed+'-tweet.json', 'r')
		feature_file = open('processed_tweets/'+feed+'-tweet.csv', 'w')
		for line in data_File_NYTimes:
			data_dict = json.loads(line)
			# print data_dict['text']
			sentence = preprocess(data_dict['text'])
			# print sentence

			sentence_vector = []
			for word in sentence:
				try:
					tmp_vec = google_model.word_vec(word).tolist() #np.array([1,1,1,1]).tolist()
				except:
					e = sys.exc_info()[0]
					print "<p>Error: %s</p>" % e
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
			sentence_vector = sentence_vector[0]
			# print sentence_vector
			# print len(sentence_vector)
			feature_file.writelines(','.join(map(str, sentence_vector))+'\n')
			# break

		feature_file.close()
		data_File_NYTimes.close()

if __name__ == '__main__':
	main()