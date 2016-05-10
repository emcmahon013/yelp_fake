import os, codecs 
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from extract_data import extract,extract_LIWC, extract_skew
from extract_watson_data import create_watson_db
import string, gensim
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer 
from sklearn.externals import joblib 

#stemmer = EnglishStemmer()
#STOPWORDS = set((stemmer.stem(w)) for w in stopwords.words('english'))
"""
	extracting feature set
	the final feature set uses bigrams, LIWC, and watson
"""

def bigram_plus(corpus,ngrams,save,polarity):
	dictionary = []
	reviews = []
	if type(ngrams)== tuple:
		bigram_vectorizer = CountVectorizer(ngram_range=ngrams,min_df=1,lowercase=False)
	elif type(ngrams) == int:
		bigram_vectorizer = CountVectorizer(min_df=1,lowercase=False)
	for r in corpus:
		reviews.append(r)
		dictionary.append(corpus[r])
	X = bigram_vectorizer.fit_transform(dictionary).toarray()
	if save == True:
		model_name = './models/'+str(polarity)+'_bigram.pkl'
		joblib.dump(bigram_vectorizer,model_name)
	return X, reviews

def tf_idf(X):
	transformer = TfidfTransformer()
	t = transformer.fit_transform(X)
	X = t.toarray()
	return X

def remove_stopwords(words):
	return [ w for w in words if w not in STOPWORDS and len(w) >=2 ]

def transform_word(word):
	return stemmer.stem(word.lower().strip(string.punctuation))

def transform_text(text):
	words = text.split()
	return remove_stopwords(map(transform_word,words))

def text_iter(corpus):
	for review in corpus:
		yield (review, corpus[review], transform_text(corpus[review]))
	
def LSI(files):
	reviews, raw_texts, texts = zip(*list(text_iter(files)))
	dictionary = gensim.corpora.dictionary.Dictionary(texts)
	dictionary.compactify()
	corpus = map(dictionary.doc2bow, texts)
	#lsi = gensim.models.lsimodel.LsiModel(corpus,id2word=dictionary,num_topics=1000)
	U, s = gensim.models.lsimodel.stochastic_svd(corpus,300,500)
	print(U)
	#print('lsi')
	#n = len(texts)
	#for i in range(n):
	#	doc = dictionary.doc2bow(texts[i])
	#	lsi_doc = lsi[doc]
		#print(gensim.matutils.corpus2dense(lsi_doc, num_terms=1000))


def add_LIWC(polarity,r_order,col_name='Filename'):
	data = extract_LIWC(polarity)
	reviews = pd.DataFrame(r_order,columns=[col_name])
	LIWC = reviews.merge(data,'left')
	#LIWC.reset_index(inplace=True)
	return LIWC 

def add_watson(polarity, feat_data):
	#watson_df = create_watson_db(polarity)
	filename = str(polarity)+'_watson.csv'
	watson_df = pd.read_csv(filename,sep=',')
	watson = feat_data.merge(watson_df, on='Filename',how='outer')
	return watson

def add_y(y):
	output = pd.DataFrame(y)
	output = output.transpose()
	output.columns = ['y_rating','fold']
	output['Filename'] = output.index
	return output 


def features(polarity,version='ott',skew=False,ngrams=(1,2),save=False):
	if skew == False:
		corpus, y = extract(polarity)
	else:
		corpus, y = extract_skew(polarity)
	X_ngram, r_order = bigram_plus(corpus,ngrams=ngrams,save=save,polarity=polarity)
	#X_tfidf = tf_idf(X_ngram)
	feats = add_LIWC(polarity,r_order)
	if version!= 'ott':
		feats = add_watson(polarity,feats)
	output= add_y(y)
	test = np.hstack((np.matrix(feats),X_ngram))
	X = pd.DataFrame(X_ngram)
	feat_data = feats.merge(X, left_index=True,right_index=True)
	if np.matrix(feat_data).all()!=test.all():
		print('data does not line up')
		print(np.shape(data))
		print(np.shape(test))
		return 
	data = feat_data.merge(output,'left')
	if len(X) != len(data):
		return 
	y = np.squeeze(np.asarray(data[['y_rating','fold']])).astype(int)
	return data, y



if __name__ =="__main__":
	corpus, y = extract('positive')
	LSI(corpus)
	#data, y = features('positive',version='watson')
	#data, y = features('negative',version='watson')
	#corpus = ['I stay at this hotel 2 times a year on business and LOVE it! The staff are great, the rooms are spacious and clean, and the location is perfect for shopping and dining on Michigan Ave. Plus if you sign up for Omni Select Membership (for free) you get free wireless internet access. ', 'This a great property, excellent location and wonderful staff. Everyone was very accommodating and polite. The room I had was on the 23rd floor and was like a suite, with a living area and a bedroom. The living room was spacious, with a plasma TV, a desk and a couch. The beds were very comfortable and the toiletries of very good quality. In the closed they placed an umbrella, which came in handy, it rained the whole time I was in Chicago. The internet connection is $9.95/24hrs. Great place, I will return for sure. ', 'This hotel is the perfect location for downtown Chicago shopping. The only thing is the pool is extremely small - it is indoors, but looks much larger on the website. ', 'The Omni is in a fabulous location on Michigan Avenue. Within just blocks are all types of stores, including Saks, Nordstroms, H&M, Filenes Basement, Macys, La Perla, Apple, Bloomingdates.....I could go on and on! The room itself was fabulous. Comfortable, nice big flat screen tvs, nice sized bathroom. They charge for Wi-Fi, but we found if we clicked yes on joining their guest program we could then go from the sign-on screen right to our email without actually completing the registration. We got this hotel for $214/night through Priceline and felt it was a terrific deal! ', 'The Omni Chicago really delivers on all fronts, from the spaciousness of the rooms to the helpful staff to the prized location on Michigan Avenue. While this address in Chicago requires a high level of quality, the Omni delivers. Check in for myself and a whole group of people with me was under 3 minutes, the staff had plentiful recommendations for dining and events, and the rooms are some of the largest you\'ll find at this price range in Chicago. Even the "standard" room has a separate living area and work desk. The fitness center has free weights, weight machines, and two rows of cardio equipment. I shared the room with 7 others and did not feel cramped in any way! All in all, a great property! ']
	#X_y = bigram_plus(corpus)
	#print(np.shape(X_y))
