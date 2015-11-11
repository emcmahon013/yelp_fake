import os, codecs 
import numpy
import pandas
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from extract_data import extract 

def bigram_plus(corpus,y_test):
	bigram_vectorizer = CountVectorizer(ngram_range=(1,2),
		token_pattern=r'\bw+\b',stop_words=stopwords.words('english'),
		min_df=1)
	# bigram = bigram_vectorizer.build_analyzer()
	# for doc in corpus:
	# 	bigram(doc)
	X_y = bigram_vectorizer.fit_transform(corpus,y_test).toarray()
	print(bigram_vectorizer.stop_words_)
	return X_y

def tf_idf(X_y):
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(X_y)
	X_y = tfidf.toarray()
	return X_y
