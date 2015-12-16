import os, codecs 
import  numpy as np
import pandas as pd
import nltk
from extract_data import extract, extract_skew
from feature_extract import features
from pred import fake_pred
from scipy.stats import itemfreq

def main(version,method,polarity,ngrams,save=False,fold=5):
	data, y = features(polarity,version=version,ngrams=ngrams,skew=False,save=save)
	print('data: '+str(np.shape(data)))
	fp = fake_pred(polarity,version=version,save=save,fold=fold)
	results, acc = fp.ott_1fold(data,method)
	for key in acc:
		print(str(version)+' / '+str(polarity)+' accuracy for '+str(key)+': '+str(acc[key]))


if __name__ == "__main__":
	polarities = ['negative','positive']
	for polarity in polarities:
		main('ott','NB',polarity,ngrams=(1,2),save=False,fold=5)
	#polarities = ['negative','positive']
	#for polarity in polarities:
	#	data, y = features(polarity,version='watson')
	#	print(data)
		#print('data: '+str(np.shape(data)))
		#fp = fake_pred(version='Ott')
		#results, acc = fp.ott_1fold(data,'NB')
		#for key in acc:
		#	print(str(polarity)+' accuracy for '+str(key)+': '+str(acc[key]))
