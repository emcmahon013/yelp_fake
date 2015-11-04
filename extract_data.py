import os
import numpy
import pandas

def extract(version):
	polarity = str(version)+'_polarity'
	os.chdir("./op_spam_v1.4/" + str(polarity))
	for f in sorted(os.listdir('.')):
		if 'truthful' in f:
			os.chdir("./"+str(f))
			print(os.getcwd())
		elif 'deceptive' in f:
			os.chdir("./"+str(f))
			print(os.getcwd())			


if __name__ == "__main__":
	extract('positive')
