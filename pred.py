import numpy as np 
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import itemfreq
from sklearn.metrics import classification_report, accuracy_score


class fake_pred:
	def __init__(self,version='Ott',fold=5,prob=False):
		self.version = version
		self.fold = fold
		self.prob = prob 


	def X_y(self,X):
		n=np.shape(X)[0]
		folds = np.squeeze(np.asarray(X[:,-1])).astype(int)
		y = np.squeeze(np.asarray(X[:,-2])).astype(int)
		X = X[:,:-2].astype(float)	
		return X, y, folds

	def sample(self,X,y,folds):
		n = np.shape(X)[0]
		lpl = cross_validation.LeavePLabelOut(folds,p=1)
		if self.fold == None:
			return lpl
		else:
			for train_index, test_index in lpl:
				if folds[test_index[0]] == self.fold:
					X_train, X_test = X[train_index], X[test_index]
					y_train, y_test = y[train_index], y[test_index]
			return X_train, X_test, y_train, y_test


	def tune(self,X_train,y_train,method):
		model={}
		if method == 'SVM':
			#parameters to tune model
			tuned_parameters = [{'kernel':['rbf'],'gamma':[1e-1,1e-3,1e-5],'C':[1,10,100]},
					{'kernel':['linear'],'C':[1,100],'gamma':[1e-3]}]
			#use gridsearch to find best model
			clf = GridSearchCV(SVC(probability=self.prob),tuned_parameters)
			clf.fit(X_train,y_train)
			print(clf.best_estimator_)
			model['svm']=clf
		elif method == 'NB':
			gnb=GaussianNB()
			gnb.fit(X_train,y_train)
			model['gnb'] = gnb
			bern = BernoulliNB()
			bern.fit(X_train,y_train)
			model['bern'] = bern
		elif method == 'LN':
			ln = LogisticRegression()
			ln.fit(X_train,y_train)
			model['log'] = ln
		elif method == 'ADT':
			bdt = {}
			bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
				n_estimators=600,
				learning_rate=1.5,
				algorithm="SAMME")
			bdt_discrete.fit(X_train,y_train)
			# bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
			# 	n_estimators=600,
			# 	learning_rate=1)
			# bdt_real.fit(X_train,y_train)
			model['bdt_discrete'] = bdt_discrete
			# model['bdt_real'] = bdt_real
		return model

	def pred(self,model,X_test,y_test):
		d={}
		d['y_test']=y_test
		for m in model:
			mod=model[m]
			d[m]=mod.predict(X_test)
			if self.prob == True:
				probas = mod.predict_proba(X_test)
				values = np.unique(d[m])
				n = np.shape(probas)[1]
				for i in range(n):
					prob_name = str(m)+'_prob_'+str(i)
					d[prob_name] = probas[:,i]
		return d

	def accuracy(self,d):
		print('y_true:\n'+str(itemfreq(d['y_test'])))
		for key in d:
			if 'prob' not in key and key!='y_test':
				y_test = d['y_test']
				y_pred = d[key]
				acc = accuracy_score(y_test,y_pred)
				print(str(key)+' accuracy: '+str(acc))

	def main(self,data,method):
		X, y, folds = self.X_y(data)
		X_train, X_test, y_train, y_test = self.sample(X,y,folds)
		model = self.tune(X_train,y_train,method)
		results = self.pred(model,X_test,y_test)
		self.accuracy(results)
		return results



