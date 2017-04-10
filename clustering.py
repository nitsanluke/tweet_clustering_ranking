from sklearn.cluster import KMeans
import sys
import json
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt

feature_data = open(sys.argv[1], 'r')
tweet_data 	 = open(sys.argv[2], 'r')

X = []
for line in feature_data:
	X.append(line.split(','))

X = np.array(X)
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters,
				random_state=0, n_init=1, init='k-means++',
				verbose=1, max_iter=1).fit(X)

print "cluster centers:\n", kmeans.cluster_centers_
cluster_file = open('cluster_centres.csv', 'w')
for i in range(num_clusters):
	cluster_file.writelines(','.join(map(str, kmeans.cluster_centers_[i].tolist())))

print "cluster inertia: \n", kmeans.inertia_
cluster_labels = kmeans.labels_
#del X

idx = 0
tweets = []
for line in tweet_data:
	data_dict = json.loads(line)
	tweets.append(data_dict['text'])

tweets = np.array(tweets)
cluster_counts = {}
for clus in range(num_clusters):
	index = np.where(cluster_labels == clus)[0]
	# cluster_counts[clus] = len(index)
	# print index
	print "\ntweets cluster: ", clus, ' count: ', len(index), '\n'
	print tweets[index[0:5]]



Y = cluster_labels
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
print X[1:5, :]
print Y[1:5]

plt.scatter(X[:, 0], X[:, 1], color='w')
plt.scatter(X[(Y == 1), 0], X[(Y == 1), 1], color='r')
plt.scatter(X[(Y == 0), 0], X[(Y == 0), 1], color='b')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()