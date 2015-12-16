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
import json


class predict_prior:
	def __init__(self,filename,model,up_to=None):
		self.model = model
		self.up_to = up_to
		self.data = pd.read_csv(filename,sep=',')
		self.hotel_prior, self.hotel_predict = self.split()


	def split(self):
		hotels = self.data 
		hotels['date'] = hotels['date'].astype('datetime64')
		hotel_prior = hotels[hotels['date'].apply(lambda x: x.year<=2014 and x.month<10)]
		hotel_predict = hotels[hotels['date'].apply(lambda x: x.year>=2014 and x.month>=10)]
		return hotel_prior, hotel_predict


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

	def prior_db(self,hotel_pred,prior):
		years = ["all",2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
		if self.up_to != None:
			years.index(up_to)
			years = years[:up_to]
		cols = ["business_id","value"]
		cols.extend(years)

		print(hotel_pred)

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
				#if r == "positive" or r == "negative":
				#	print (b, hotel_row[2])
		print(hotel_priors)


		hotel_star = hotel_priors[hotel_priors["value"]=="adj_stars"]
		adj_star = hotel_star[["business_id","all"]]
		adj_star.rename(columns={'all':'adj_stars'},inplace=True)
		mu = np.mean(hotel_star["all"])
		var = np.var(hotel_star["all"])
		adj_star.set_index('business_id')
		ranking = norm.cdf(hotel_star["all"],loc=mu,scale=var)
		adj_star['ranking'] = ranking
		pos_perc = hotel_priors[hotel_priors["value"]=="positive"][['business_id','all']]
		neg_perc = hotel_priors[hotel_priors["value"]=="negative"][['business_id','all']]
		pos_perc.rename(columns={'all':'positive'},inplace=True)
		neg_perc.rename(columns={'all':'negative'},inplace=True)
		first = adj_star.merge(pos_perc,on="business_id")
		ratings = first.merge(neg_perc,on="business_id")
		bus_df = pd.DataFrame(self.data.groupby('business_id')[['b_name','b_city']].max())
		business_df = bus_df.merge(ratings,right_on='business_id',left_index=True)
		
		if prior == False:
			business_df.to_csv('business_data.csv',sep=',',index=False)
			hotel_priors.to_csv('hotels_data.csv',sep=',',index=False)
		else:
			hotel_priors.to_csv('hotels_prior.csv',sep=',',index=False)


	def compute_prior(self,prior=True):
		hotel_pred = {}

		col_names = list(self.hotel_prior.keys())
		WC = col_names.index('WC')
		pos_bigram = self.load_model('positive',bigram=True)
		pos_model = self.load_model('positive')
		neg_bigram = self.load_model('negative',bigram=True)
		neg_model = self.load_model('negative')
		if prior == True:
			n = len(self.hotel_prior)
		else:
			n = len(self.data)
		for i in range(n):
			if prior == True:
				hotel = self.hotel_prior.iloc[i]
			else:
				hotel = self.data.iloc[i]

			star = hotel['stars']
			text = hotel['text']
			business_id = hotel['business_id']
			date = hotel['date']
			text_len = len(text.split())

			try:
				hotel_pred[business_id]
			except KeyError:
				hotel_pred[business_id]={"all":{"positive":[],"negative":[],"adj_stars":[],"stars":[],"pos_count":0,"neg_count":0,"total_count":0}}
				#hotel_pred[business_id]={"positive":{"all":[]},"negative":{"all":[]},"adj_stars":{"all":[]},"stars":{"all":[]},"count":{"all":0}}

			if (star == 4 or star == 5) and text_len > 50:
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

			elif (star == 1 or star == 2) and text_len > 50:
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

		self.prior_db(hotel_pred,prior)



	def by_hotel(self):
		reviews = {}
		business = pd.read_csv('business_data.csv',sep=',')
		business.set_index('business_id',inplace=True)
		hotel_ts = pd.read_csv('hotels_data.csv',sep=',')


		n = len(self.hotel_predict)
		col_names = list(self.hotel_predict.keys())
		WC = col_names.index('WC')
		pos_bigram = self.load_model('positive',bigram=True)
		pos_model = self.load_model('positive')
		neg_bigram = self.load_model('negative',bigram=True)
		neg_model = self.load_model('negative')

		m = len(hotel_ts)
		ts = {}
		dates = list(hotel_ts.keys())[3:]
		for j in range(m):
			t = hotel_ts.iloc[j]
			b_id = t['business_id']
			ts[b_id] = {"positive_ts":{"date":dates,"value":[]},"negative_ts":{"date":dates,"value":[]},"adj_star":{"date":[],"value":[]}}
			if t['value'] == 'positive':
				ts[b_id]['positive_ts'] = list(t[3:])
			if t['value'] == 'negative':
				ts[b_id]['negative_ts'] = list(t[3:])
			if t['value'] == 'adj_star':
				ts[b_id]['adj_star']= list(t[3:])

			reviews[b_id]={}
			reviews[b_id]["name"] = business['b_name'][b_id]
			reviews[b_id]["city"] = business['b_city'][b_id]
			reviews[b_id]["star"] = str(round(business['adj_stars'][b_id],2))
			rounded_stars = round(business['adj_stars'][b_id]*2)/2
			reviews[b_id]["rounded_star"] = str(rounded_stars)
			reviews[b_id]["image_url"] = business['image_url'][b_id]
			reviews[b_id]["rank"] = str(round(business['ranking'][b_id]*100))
			reviews[b_id]["positive"] = str(round(business['positive'][b_id]*100))+'%'
			reviews[b_id]["negative"] = str(round(business['negative'][b_id]*100))+'%'

		for i in range(n):
			hotel = self.hotel_predict.iloc[i]
			b_id = hotel['business_id']
			r_name = hotel['u_name']
			r_star = str(float(hotel['stars']))
			r_date = str(hotel['date'].year)+'-'+str(hotel['date'].month)+'-'+str(hotel['date'].day)
			r_text = hotel['text']

			r_type = "Truthful"
			if (r_star == '5.0' or r_star == '4.0') and len(r_text.split()) > 50:
				prob, r_true = self.get_prediction(r_text,pos_bigram,pos_model,hotel,col_names,WC)
				r_prob = str(round(float(prob[0][1])))
				if r_true == 1:
					r_type = "Positive Deceptive"
			elif (r_star == '1.0' or r_star == '2.0') and len(r_text.split()) > 50:
				prob, r_true = self.get_prediction(r_text,neg_bigram,neg_model,hotel,col_names,WC)
				r_prob = str(round(float(prob[0][1])))
				if r_true == 1:
					r_type = "Negative Deceptive"
			else:
				r_prob = "NA"

			review_dict = {"name":r_name,"star":r_star,"date":r_date,"review":r_text,"type":r_type,"probability":r_prob}
			try:
				reviews[b_id]["reviews"].append(review_dict)
			except KeyError:
				reviews[b_id]["reviews"] = [review_dict]

		return reviews 

if __name__ =="__main__":
	PPrior = predict_prior('hotel_yelp_reviews.csv','multi')
	#PPrior.compute_prior(prior=False)
	reviews = PPrior.by_hotel()
	print(reviews)
	with open('./application/data/reviews.json','w') as outfile:
		json.dump(reviews,outfile)

