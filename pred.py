import numpy as np 
import pandas as pd 
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import itemfreq
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from sklearn.externals import joblib 
import pickle
from datetime import datetime
from scipy.stats import norm 

class fake_pred:
	def __init__(self,polarity,version='Ott',fold=5,prob=False,save=False):
		self.polarity = polarity
		self.version = version
		self.fold = fold
		self.prob = prob 
		self.save = save
		


	def X_y(self,X):
		n=np.shape(X)[0]
		folds = np.squeeze(np.asarray(X[:,-1])).astype(int)
		y = np.squeeze(np.asarray(X[:,-2])).astype(int)
		X = X[:,:-2].astype(float)
		return X, y, folds

	def sample(self,X,y,folds):
		n = np.shape(X)[0]
		self.lpl = cross_validation.LeavePLabelOut(folds,p=1)
		if self.fold == 'all':
			X_train = X
			y_train = y
			X_test = X
			y_test = y 
		else:
			for train_index, test_index in self.lpl:
				if folds[test_index[0]] == self.fold:
					X_train, X_test = X[train_index], X[test_index]
					y_train, y_test = y[train_index], y[test_index]
					self.train_folds, self.test_folds = folds[train_index], folds[test_index]
		return X_train, X_test, y_train, y_test			


	def tune(self,X_train,y_train,method,folds,class_prior):
		model={}
		if method == 'SVM':
			#parameters to tune model
			tuned_parameters = [{'kernel':['rbf'],'gamma':[1e-1,1e-3,1e-5],'C':[1,10,100]},
					{'kernel':['linear'],'C':[1,100],'gamma':[1e-3]}]
			#use gridsearch to find best model
			clf = GridSearchCV(SVC(probability=self.prob),tuned_parameters)
			clf.fit(X_train,y_train)
			#print(clf.best_estimator_)
			model['svm']=clf
		elif method == 'NB':
			# gnb=GaussianNB()
			# gnb.fit(X_train,y_train)
			# model['gnb'] = gnb
			# bern = BernoulliNB()
			# bern.fit(X_train,y_train)
			# model['bern'] = bern
			multi = MultinomialNB(class_prior=class_prior)
			multi.fit(X_train,y_train)
			print('priors: '+str(multi.class_log_prior_))
			model['multi'] = multi
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
			if self.save == True:
				model_name = './models/'+str(self.polarity)+'_'+str(m)+'.pkl'
				joblib.dump(model[m],model_name)
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
		acc = {}
		for key in d:
			if 'prob' not in key and key!='y_test':
				y_test = d['y_test']
				y_pred = d[key]
				score = accuracy_score(y_test,y_pred)
				acc[key] = score
		return acc

	def ott_1fold(self,data,method,class_prior=None):
		X, y, folds = self.X_y(data)
		X_train, X_test, y_train, y_test = self.sample(X,y,folds)
		model = self.tune(X_train,y_train,method,folds,class_prior)
		results = self.pred(model,X_test,y_test)
		acc = self.accuracy(results)
		return results, acc

	def ott_5fold(self,data,method,class_prior=None):
		folds = [1,2,3,4,5]
		total_acc = {}
		for f in folds:
			self.fold = f
			results, acc = self.ott_1fold(data,method,class_prior=class_prior)
			for key in acc:
				try:
					total_acc[key]+= acc[key]
				except:
					total_acc[key] = acc[key]
		for key in total_acc:
			total_acc[key] = total_acc[key]/len(folds)
		return results, total_acc


	def watson_main(self,data,method):
		X, y, folds = self.X_y(data)
		X_train, X_test, y_train, y_test = self.sample(X,y,folds)






if __name__ =="__main__":
	pred = fake_pred(positive)
	results, total_acc = ott_5fold('positive')









