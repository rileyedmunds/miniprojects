#Recommender Systems:
	#Collaborative: prediction based on what others liked
	#Content-based: prediction based on what you've liked
	#Collaborative: uses both

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM 

#gather data and parse it
data = fetch_movielens(min_rating=4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#model creation (uses SGD on weighted approximate rank pairwise)
model = LightFM(loss='warp')
#training
model.fit(data['train'], epochs=50, num_threads=2)

def sample_recommendation(model, data, user_ids):
	#number of users and movies in training data
	u_users, n_movies = data['train'].shape

	#generate recommendations for each user
	for user_id in user_ids:
		#positives (5 is positive, below 5 is negative)
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#recommendations from movies known:
		scores = model.predict(user_id, np.arange(n_movies))
		#rank in order (descending rating)
		top_items = data['item_labels'][np.argsort(-scores)]

		#print results:
		print("User %s" % user_id)

		print("   We know you like:")
		for x in known_positives[:3]:
			print("                %s" % x)
		print("   Recommendations:")
		for x in top_items[:3]:
			print("                %s" % x)


sample_recommendation(model, data, [3,25,450])