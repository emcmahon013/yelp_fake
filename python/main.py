import os, codecs 
import  numpy as np
import pandas as pd
import nltk
from extract_data import extract, extract_skew
from feature_extract import features
from pred import fake_pred
from scipy.stats import itemfreq

if __name__ == "__main__":
	polarities = ['negative','positive']
	for polarity in polarities:
		data, y = features(polarity)
		#print('data: '+str(np.shape(data)))
		fp = fake_pred(version='Ott')
		results, acc = fp.ott_1fold(data,'NB')
		for key in acc:
			print(str(polarity)+' accuracy for '+str(key)+': '+str(acc[key]))
