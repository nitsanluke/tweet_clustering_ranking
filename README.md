# Cluster and Rank Tweets

These scripts download tweets from list of predifined news feeds and cluster them based on similariy of word2vec features and rank most influential tweets. Essentially an approch to filter relavant tweets

Eg:
Tweets from 9 news feeds ('nytimes', 'thesun', 'thetimes', 'ap', 'cnn’, 'bbcnews', 'cnet', 'msnuk', 'telegraph') are download every 10mins. Every tweet is preprocessed and the tweet text is transformed into vectors via word2vec model. 
Then every tweet is classified into a cluster and then all tweets belonging to the cluster are ranked based on a score computed from other tweet features (favorited, etc). Once the top tweet for every cluster is determined it’s stored into a dictionary and if a tweet score exceeds a predefined threshold it will be written into a cluster-specific file.

**Run script: python live_processing_app.py GoogleNews-vectors-negative300.bin**

## Clustering Tweets

Tweets from the 9 feeds were downloaded initially and every tweet text is transformed into a 7,500 dimension vector using the word2vec model. Each word is represented as a 300-dimensional numeric vector. Once the features are extracted a simple K-means algorithm is used to cluster the tweets and based on the experiments with historical tweets 3 clusters seemed appropriate. 

## Ranking Tweets

From the tweets downloaded a Random Forest regression model is trained using the retweet count as a proxy for the importance of a tweet (as a first attempt to quantify the importance of a tweet). This model use other features apart from the tweet text, such as ‘favorited',  'retweeted_status',  'retweeted',  'entities',  'favorite_count',  'possibly_sensitive'. The importance of a tweet is a hard to define phenomenon and it is influenced by many other factors than the tweets itself but the regression model seems to show promising results in actually being able to capture the retweet count trend at least when it’s positive and negative. From the positive results In future, this can be extended to predict future retweet count of a tweet by tracking tweet retweet count and generating a labeled dataset.
 

## Other Source Flies

read_tweets.py - Gives a simple way to read stored tweets from a JSON objests and compute word2vec representation of each tweet text

dowload_tweets.py - Gives a simple way to download tweets from the twitter API and store them in files as JSON objects

*_test.py - samples to test the tweepy  API and the google word2vec model
