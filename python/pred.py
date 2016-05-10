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


"""
	class to create predictions
	- predictions segmented by positive or negative polarity 
	- ability to sample 5-fold or random
	- can run SVM, Naive Bayes, logistic, or AdaBoosted DTs
"""
class fake_pred:
	def __init__(self,polarity,version='Ott',fold=5,prob=False,save=False,prior=None):
		self.polarity = polarity
		self.version = version
		self.fold = fold
		self.prob = prob 
		self.save = save
		self.prior = None
		
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

	def X_y(self,data,output,sample=None,skew=None):
		if sample == None:
			X = np.matrix(data)[:,1:]
			n=np.shape(X)[0]
			folds = np.squeeze(np.asarray(data['fold'])).astype(int)
			y = np.squeeze(np.asarray(data['y_rating'])).astype(int)
			X = X[:,:-2].astype(float)
			X_train, X_test, y_train, y_test = self.sample(X,y,folds)
		elif sample == 'ott':
			test_full = data[data['fold']==self.fold]
			training_set = data[data['fold']!=self.fold]
			pos_test = test_full[test_full['y_rating']==1]
			neg_test = test_full[test_full['y_rating']==0]
			pos_sample = pos_test.sample(frac=skew*2)
			if skew == .5 and len(pos_sample)!=len(pos_test):
				print('sample does not line up')
				return 
			test_set = pd.concat([pos_sample,neg_test])
			X_train = (np.matrix(training_set)[:,1:-2]).astype(float)
			y_train = np.squeeze(np.asarray(training_set['y_rating'])).astype(int)
			X_test = (np.matrix(test_set)[:,1:-2]).astype(float)
			y_test = np.squeeze(np.asarray(test_set['y_rating'])).astype(int)
		elif sample == 'watson':
			training = data[data['fold']!=self.fold]
			test = data[data['fold']==self.fold]
			pos_training = training[training['y_rating']==1]
			neg_train = training[training['y_rating']==0]
			pos_set = test[test['y_rating']==1]
			neg_test = test[test['y_rating']==0]
			pos_train = pos_training.sample(frac=skew*2)
			training_set = pd.concat([pos_train,neg_train])
			pos_test = pos_set.sample(frac=skew*2)
			test_set = pd.concat([pos_test,neg_test])
			X_train = (np.matrix(training_set)[:,1:-2]).astype(float)
			y_train = np.squeeze(np.asarray(training_set['y_rating'])).astype(int)
			X_test = (np.matrix(test_set)[:,1:-2]).astype(float)
			y_test = np.squeeze(np.asarray(test_set['y_rating'])).astype(int)
		elif sample == 'prior':
			data.to_csv('test.csv')
			filename = data['Filename']
			for f in filename:
				print(f)
				p = re.compile('_\d*\.txt')
				if p.match(f):
					print(True)
				else:
					print(False)


					deceptive = random.sample(range(1,20),skew_n)
					filename = name + "/" + f + "/fold" + str(i)
					for r in sorted(os.listdir(filename)):
						skew = True
						if t == 'deceptive':
							skew = False
							for dec in deceptive:
								if str(dec)+'.txt' in r:
									skew = True




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
			#print(clf.best_estimator_)
			model['svm']=clf
		elif method == 'NB':
			# gnb=GaussianNB()
			# gnb.fit(X_train,y_train)
			# model['gnb'] = gnb
			# bern = BernoulliNB()
			# bern.fit(X_train,y_train)
			# model['bern'] = bern
			multi = MultinomialNB(class_prior=self.prior)
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

	def ott_1fold(self,data,y,method,sample=None,skew=None):
		X_train, X_test, y_train, y_test = self.X_y(data,y,sample=sample,skew=skew)
		model = self.tune(X_train,y_train,method)
		results = self.pred(model,X_test,y_test)
		acc = self.accuracy(results)
		return results, acc

	def ott_5fold(self,data,y,method,sample=None,skew=None):
		folds = [1,2,3,4,5]
		total_acc = {}
		for f in folds:
			self.fold = f
			results, acc = self.ott_1fold(data,y,method,sample=sample,skew=skew)
			for key in acc:
				try:
					total_acc[key]+= acc[key]
				except:
					total_acc[key] = acc[key]
		for key in total_acc:
			total_acc[key] = total_acc[key]/len(folds)
		return results, total_acc







if __name__ =="__main__":
	pred = fake_pred(positive)
	results, total_acc = ott_5fold('positive')









