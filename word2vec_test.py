import gensim
# import logging
import nltk
# from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words('english')


#Loading Google Word2Vec model
google_model = gensim.models.KeyedVectors.load_word2vec_format('google-word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# print google_model.most_similar('frog')
# print google_model.word_vec('apple')
# print google_model['student']

sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
sentence_president = 'The president greets the press in Chicago'.lower().split()
sentence_obama = [w for w in sentence_obama if w not in stopwords]
sentence_president = [w for w in sentence_president if w not in stopwords]

wm_distance = google_model.wmdistance(sentence_obama, sentence_president)
cos_sim = google_model.n_similarity(sentence_obama, sentence_president)

print "WM Distance: ",  wm_distance
print "Cos Sim: ", cos_sim
