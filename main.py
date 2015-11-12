import os, codecs 
import  numpy as np
import pandas as pd
import nltk
from extract_data import extract
from feature_extract import bigram_plus, tf_idf, add_LIWC


if __name__ == "__main__":
	polarity = 'positive'
	pos_corpus, pos_y = extract(polarity)
	X_ngram, r_order = bigram_plus(pos_corpus)
	LIWC = add_LIWC(polarity,r_order,pos_y)
	X = np.hstack((X_ngram,np.matrix(LIWC)[:,1:]))
	print('X: '+str(np.shape(X)))

