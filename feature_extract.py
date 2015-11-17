import os, codecs 
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from extract_data import extract,extract_LIWC

def bigram_plus(corpus,ngrams=(1,2)):
	dictionary = []
	reviews = []
	bigram_vectorizer = CountVectorizer(ngram_range=ngrams,min_df=1)
	# bigram = bigram_vectorizer.build_analyzer()
	# for r in corpus:
	# 	bigram(corpus[r])
	for r in corpus:
		reviews.append(r)
		dictionary.append(corpus[r])
	X = bigram_vectorizer.fit_transform(dictionary).toarray()
	return X, reviews

def tf_idf(X):
	transformer = TfidfTransformer()
	t = transformer.fit_transform(X)
	X = t.toarray()
	return X

def add_LIWC(polarity,r_order,y):
	data = extract_LIWC(polarity)
	reviews = pd.DataFrame(r_order,columns=['Filename'])
	LIWC = reviews.merge(data,'left')
	output = pd.DataFrame(y)
	output = output.transpose()
	output.columns = ['y_rating','fold']
	output['Filename'] = output.index
	LIWC = LIWC.merge(output,'left')
	return LIWC 




# if __name__ =="__main__":
# 	corpus = ['I stay at this hotel 2 times a year on business and LOVE it! The staff are great, the rooms are spacious and clean, and the location is perfect for shopping and dining on Michigan Ave. Plus if you sign up for Omni Select Membership (for free) you get free wireless internet access. ', 'This a great property, excellent location and wonderful staff. Everyone was very accommodating and polite. The room I had was on the 23rd floor and was like a suite, with a living area and a bedroom. The living room was spacious, with a plasma TV, a desk and a couch. The beds were very comfortable and the toiletries of very good quality. In the closed they placed an umbrella, which came in handy, it rained the whole time I was in Chicago. The internet connection is $9.95/24hrs. Great place, I will return for sure. ', 'This hotel is the perfect location for downtown Chicago shopping. The only thing is the pool is extremely small - it is indoors, but looks much larger on the website. ', 'The Omni is in a fabulous location on Michigan Avenue. Within just blocks are all types of stores, including Saks, Nordstroms, H&M, Filenes Basement, Macys, La Perla, Apple, Bloomingdates.....I could go on and on! The room itself was fabulous. Comfortable, nice big flat screen tvs, nice sized bathroom. They charge for Wi-Fi, but we found if we clicked yes on joining their guest program we could then go from the sign-on screen right to our email without actually completing the registration. We got this hotel for $214/night through Priceline and felt it was a terrific deal! ', 'The Omni Chicago really delivers on all fronts, from the spaciousness of the rooms to the helpful staff to the prized location on Michigan Avenue. While this address in Chicago requires a high level of quality, the Omni delivers. Check in for myself and a whole group of people with me was under 3 minutes, the staff had plentiful recommendations for dining and events, and the rooms are some of the largest you\'ll find at this price range in Chicago. Even the "standard" room has a separate living area and work desk. The fitness center has free weights, weight machines, and two rows of cardio equipment. I shared the room with 7 others and did not feel cramped in any way! All in all, a great property! ']
# 	X_y = bigram_plus(corpus)
# 	print(np.shape(X_y))
