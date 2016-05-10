import re
import pandas as pd

test = pd.read_csv('test.csv')
filename = test['Filename']

p = re.compile('.*_\d*\.txt')

for f in filename:
	if p.match(f):
		print(True)
	else:
		print(False)