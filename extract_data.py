import os, codecs
import numpy as np
import pandas as pd
import random

def extract(polarity):
	corpus = {}
	y = {}
	name = './op_spam_v1.4/'+str(polarity)+'_polarity'
	for f in sorted(os.listdir(name)):
		types = ['truthful', 'deceptive']
		for t in types:
			if t in f:
				for i in range(1,6):
					filename = name + "/" + f + "/fold" + str(i) 
					for r in sorted(os.listdir(filename)):
						if '.txt' in r:
							with codecs.open(filename + "/" + r, "r", "utf-8", "ignore") as d:
								# print ("Review for", r)
								raw_text = d.read()
								text = raw_text.replace('\n','')
								f_name = str(i)+'_'+str(r)
								corpus[r] = text
								if r[0] == 't':
									y[r] = [0,i]
								elif r[0] == 'd':
									y[r] = [1,i]
								else:
									print('ERROR with test outcome.')
	return corpus, y

def extract_skew(polarity,skew_perc=.3):
	skew_n = int(skew_perc*20)
	corpus = {}
	y = {}
	name = './op_spam_v1.4/'+str(polarity)+'_polarity'
	for f in sorted(os.listdir(name)):
		types = ['truthful', 'deceptive']
		for t in types:
			if t in f:
				for i in range(1,6):
					deceptive = random.sample(range(1,20),skew_n)
					filename = name + "/" + f + "/fold" + str(i)
					for r in sorted(os.listdir(filename)):
						skew = True
						if t == 'deceptive':
							skew = False
							for dec in deceptive:
								if str(dec)+'.txt' in r:
									skew = True
						if '.txt' in r and skew == True:
							with codecs.open(filename + "/" + r, "r", "utf-8", "ignore") as d:
								# print ("Review for", r)
								raw_text = d.read()
								text = raw_text.replace('\n','')
								f_name = str(i)+'_'+str(r)
								corpus[r] = text
								if r[0] == 't':
									y[r] = [0,i]
								elif r[0] == 'd':
									y[r] = [1,i]
								else:
									print('ERROR with test outcome.')
	return corpus, y

def extract_LIWC(polarity):
	file_path = "./op_spam_v1.4/LIWC_"+str(polarity)+'.csv'
	data = pd.read_csv(file_path)
	return data