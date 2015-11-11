import os, codecs 
import  numpy as np
import pandas as pd
import nltk
from extract_data import extract
from feature_extract import bigram_plus, tf_idf


if __name__ == "__main__":
	pos_corpus, pos_test = extract('positive')
	neg_corpus, neg_test = extract('negative')
	X_y = bigram_plus(pos_corpus)
	print(np.shape(X_y),np.shape(pos_test))

