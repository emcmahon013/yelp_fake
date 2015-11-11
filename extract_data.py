import os, codecs
import numpy
import pandas

def extract(version):
	corpus = []
	y_test = []
	polarity = str(version)+'_polarity'
	os.chdir("./op_spam_v1.4/" + str(polarity))
	for f in sorted(os.listdir('.')):
		types = ['truthful', 'deceptive']
		for t in types:
			if t in f:
				for i in range(1,4): 
					filename = "./" + f + "/fold" + str(i) 
					for r in sorted(os.listdir(filename)):
						if '.txt' in r:
							with codecs.open(filename + "/" + r, "r", "utf-8", "ignore") as d:
								print ("Review for", r)
								raw_text = d.read()
								text = raw_text.replace('\n','')
								corpus.append(text)
								if r[0] == 't':
									y_test.append(1)
								elif r[0] == 'd':
									y_test.append(0)
								else:
									print('ERROR with test outcome.')
	return corpus, y_test



if __name__ == "__main__":
	pos_corpus, pos_test = extract('positive')
	neg_corpus, neg_test = extract('negative')
