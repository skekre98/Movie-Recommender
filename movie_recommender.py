import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


################################## helper functions
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    try:
	    return df[df.title == title]["index"].values[0]
    except:
        return -1
##################################################

# Read CSV File
df = pd.read_csv("iMDB.csv")

# Selecting Features
features = ['keywords','cast','genres','director']

# combine selected features
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
	return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]

df["combined_features"] = df.apply(combine_features,axis=1)


# count matrix using new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix) 

# movie in search
movie_index = -1
while movie_index == -1:
    movie_user_likes = str(input("What movie do you like? "))
    movie_user_likes = movie_user_likes.strip()
    movie_index = get_index_from_title(movie_user_likes)
    if movie_index == -1:
        print("Oops! Movie does not exist.")

# Get a list of similar movies in descending order of similarity score
similar_movies =  list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

# Print first 50 similar movies
i = 0
for element in sorted_similar_movies:
		print(get_title_from_index(element[0]))
		i += 1
		if i > 50:
			break
