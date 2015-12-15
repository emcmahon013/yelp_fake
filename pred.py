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


	def tune(self,X_train,y_train,method,folds):
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
			multi = MultinomialNB(class_prior=[.8,.2])
			multi.fit(X_train,y_train)
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

	def ott_1fold(self,data,method):
		X, y, folds = self.X_y(data)
		X_train, X_test, y_train, y_test = self.sample(X,y,folds)
		model = self.tune(X_train,y_train,method,folds)
		results = self.pred(model,X_test,y_test)
		acc = self.accuracy(results)
		return results, acc

	def ott_5fold(self,data,method):
		folds = [1,2,3,4,5]
		total_acc = {}
		for f in folds:
			self.fold = f
			results, acc = self.ott_1fold(data,method)
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

class predict_prior:
	def __init__(self,model,up_to=None):
		self.model = model
		self.up_to = up_to

	def load_model(self,polarity,bigram=False):
		if bigram == False:
			model_file = './models/'+str(polarity)+'_'+str(self.model)+'.pkl'
		else:
			model_file = './models/'+str(polarity)+'_bigram.pkl'
		model = joblib.load(model_file)
		return model

	def get_prediction(self,text,bigram,model,hotel,col_names,WC):
		X_ngram = bigram.transform([text]).toarray()
		LIWC = np.matrix(hotel[col_names[WC:]])
		X = np.hstack((X_ngram,LIWC))
		r_prob = model.predict_proba(X)
		r_true = model.predict(X)
		return r_prob, r_true 

	def compute_prior(self,filename):
		hotel_pred = {}

		hotels = pd.read_csv(filename,sep=',')
		col_names = list(hotels.keys())
		WC = col_names.index('WC')
		pos_bigram = self.load_model('positive',bigram=True)
		pos_model = self.load_model('positive')
		neg_bigram = self.load_model('negative',bigram=True)
		neg_model = self.load_model('negative')
		n = len(hotels)
		for i in range(n):
			hotel = hotels.iloc[i]
			star = hotel['stars']
			text = hotel['text']
			business_id = hotel['business_id']
			date = datetime.strptime(hotel['date'],"%Y-%m-%d")
			text_len = len(text.split())

			try:
				hotel_pred[business_id]
			except KeyError:
				hotel_pred[business_id]={"all":{"positive":[],"negative":[],"adj_stars":[],"stars":[],"pos_count":0,"neg_count":0,"total_count":0}}
				#hotel_pred[business_id]={"positive":{"all":[]},"negative":{"all":[]},"adj_stars":{"all":[]},"stars":{"all":[]},"count":{"all":0}}

			if star == 5 and text_len > 100:
				r_prob, r_true = self.get_prediction(text,pos_bigram,pos_model,hotel,col_names,WC)
				hotel_pred[business_id]["all"]["positive"].append(float(r_true))
				hotel_pred[business_id]["all"]["pos_count"] += 1 
				hotel_pred[business_id]["all"]["stars"].append(star)
				hotel_pred[business_id]["all"]["total_count"] += 1
				try:
					hotel_pred[business_id][date.year]
				except KeyError:
					hotel_pred[business_id][date.year]={"positive":[],"negative":[],"adj_stars":[],"stars":[],"pos_count":0,"neg_count":0,"total_count":0}
				hotel_pred[business_id][date.year]["positive"].append(float(r_true))
				hotel_pred[business_id][date.year]["pos_count"] += 1
				hotel_pred[business_id][date.year]["stars"].append(star)
				hotel_pred[business_id][date.year]["total_count"] += 1
				if r_true == 0:
					hotel_pred[business_id][date.year]["adj_stars"].append(star)
					hotel_pred[business_id]["all"]["adj_stars"].append(star)

			elif star == 1 and text_len > 100:
				r_prob, r_true = self.get_prediction(text,neg_bigram,neg_model,hotel,col_names,WC)
				hotel_pred[business_id]["all"]["negative"].append(float(r_true))
				hotel_pred[business_id]["all"]["neg_count"] += 1 
				hotel_pred[business_id]["all"]["stars"].append(star)
				hotel_pred[business_id]["all"]["total_count"] += 1
				try:
					hotel_pred[business_id][date.year]
				except KeyError:
					hotel_pred[business_id][date.year]={"positive":[],"negative":[],"adj_stars":[],"stars":[],"pos_count":0,"neg_count":0,"total_count":0}
					
				hotel_pred[business_id][date.year]["negative"].append(float(r_true))
				hotel_pred[business_id][date.year]["neg_count"] += 1
				hotel_pred[business_id][date.year]["stars"].append(star)
				hotel_pred[business_id][date.year]["total_count"] += 1
				if r_true == 0:
					hotel_pred[business_id][date.year]["adj_stars"].append(star)
					hotel_pred[business_id]["all"]["adj_stars"].append(star)
			else:
				hotel_pred[business_id]["all"]["stars"].append(star)
				hotel_pred[business_id]["all"]["adj_stars"].append(star)
				hotel_pred[business_id]["all"]["total_count"] += 1
				try:
					hotel_pred[business_id][date.year]
				except KeyError:
					hotel_pred[business_id][date.year] = {"positive":[],"negative":[],"adj_stars":[],"stars":[],"pos_count":0,"neg_count":0,"total_count":0}

				hotel_pred[business_id][date.year]["stars"].append(star)
				hotel_pred[business_id][date.year]["adj_stars"].append(star)
				hotel_pred[business_id][date.year]["total_count"] += 1

		years = ["all",2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
		if self.up_to != None:
			years.index(up_to)
			years = years[:up_to]
		cols = ["business_id","value"]
		cols.extend(years)
		hotel_priors = pd.DataFrame(columns=cols)
		rows = ['positive','pos_count','negative','neg_count',"adj_stars","stars","total_count"]
		for b in hotel_pred:
			d = dict(zip(rows,[None]*len(rows)))
			for year in years:
				if year == "all":
					d['positive'] = [np.mean(hotel_pred[b][year]["positive"])]
					d['negative'] = [np.mean(hotel_pred[b][year]["negative"])]
					d['adj_stars'] = [np.mean(hotel_pred[b][year]["adj_stars"])]
					d['stars'] = [np.mean(hotel_pred[b][year]["stars"])]
					d['pos_count'] = [hotel_pred[b][year]["pos_count"]]
					d['neg_count'] = [hotel_pred[b][year]["neg_count"]]
					d['total_count'] = [hotel_pred[b][year]["total_count"]]
				else:
					for r in rows:
						try:
							if r == 'positive':
								d[r].append(np.mean(hotel_pred[b][year]["positive"]))
							elif r == 'negative':
								d[r].append(np.mean(hotel_pred[b][year]["negative"]))
							elif r == 'adj_stars':
								d[r].append(np.mean(hotel_pred[b][year]["adj_stars"]))
							elif r == 'stars':
								d[r].append(np.mean(hotel_pred[b][year]["stars"]))
							elif r == 'pos_count':
								d[r].append(hotel_pred[b][year]["pos_count"])
							elif r == 'neg_count':
								d[r].append(hotel_pred[b][year]["neg_count"])
							elif r == 'total_count':
								d[r].append(hotel_pred[b][year]["total_count"])
						except KeyError:
							d[r].append(np.nan)
			for r in rows:
				hotel_row = [str(b), str(r)] + d[r]
				hotel_priors.loc[str(b)+str(r)] = hotel_row

		return hotel_priors 


	def by_hotel(self,filename):
		reviews = {}
		hotels = pd.read_csv(filename)
		n = len(hotels)
		col_names = list(hotels.keys())
		WC = col_names.index('WC')
		pos_bigram = self.load_model('positive',bigram=True)
		pos_model = self.load_model('positive')
		neg_bigram = self.load_model('negative',bigram=True)
		neg_model = self.load_model('negative')
		time_series = pd.read_csv('hotels_priors.csv')
		m = len(time_series)
		ts = {}
		for j in range(m):
			t = time_series.iloc[j]
			b_id = t['business_id']
			ts[b_id] = {"positive_ts":[],"negative_ts":[],"positive":0,"negative":0,"ranking":0}
			if t['polarity'] == 'positive':
				ts[b_id]['positive_ts'] = list(t[3:])
				ts[b_id]['positive'] = t['total'] 
			if t['polarity'] == 'negative':
				ts[b_id]['negative_ts'] = list(t[3:])
				ts[b_id]['negative'] = t['total']
			ts['ranking'] = t['ranking']

			try:
				reviews[business_id]
			except KeyError:
				reviews[business_id] = {}
				reviews[business_id]["name"] = b_name
				reviews[business_id]["city"] = b_city
				reviews[business_id]["star"] = star
				reviews[business_id]["image_url"] = "https://www.visitsitaly.com/tours/campania/rentals/hotel_santa_caterina/hotel_santa_caterina_pool.jpg"
				reviews[business_id]["rank"] = rank



		for i in range(n):
			hotel = hotels.iloc[i]
			business_id = hotel['business_id']
			b_name = hotel['b_name']
			b_city = hotel['b_city']
			rank = ts[business_id]['ranking']
			star = 'NA'


			reviews[business_id]["reviews"] = []
			r_name = hotel['u_name']
			r_star = hotel['stars']
			r_date = hotel['date']
			r_text = hotel['text']
			r_type = "Truthful"
			if r_star == 5:
				r_prob, r_true = self.get_prediction(text,pos_bigram,pos_model,hotel,col_names,WC)
				if r_true == 1:
					r_type = "Positive Deceptive"
			elif r_star == 1:
				r_prob, r_true = self.get_prediction(text,neg_bigram,neg_model,hotel,col_names,WC)
				if r_true == 1:
					r_type = "Negative Deceptive"
			else:
				r_prob = "NA"

			review_dict = {"name":r_name,"stars":r_star,"date":r_date,"review":r_text,"type":r_type,"probability":r_prob}
			reviews[business_id]["reviews"].append(review_dict)


		return reviews 


			




if __name__ =="__main__":
	PPrior = predict_prior('multi')
	hotel_priors = PPrior.compute_prior('hotel_yelp_reviews_prior.csv')
	hotel_priors.to_csv('hotels_prior_09_2014.csv',sep=',')








