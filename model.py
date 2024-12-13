import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np


# Milestone 4

print("\n\n\n")



def create_X(df):
    # Step 1: Get the number of unique users and movies
    M = df['user_id'].nunique()  # Number of unique users
    N = df['movie_id'].nunique()  # Number of unique movies

    # Step 2: Create mapping dictionaries for users and movies
    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movie_id"]), list(range(N))))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movie_id"])))

    # Step 3: Map user IDs and movie IDs to their corresponding matrix indices
    user_index = [user_mapper[i] for i in df['user_id']]
    item_index = [movie_mapper[i] for i in df['movie_id']]

    # Step 4: Create the sparse matrix
    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))
    
    return X, movie_mapper, movie_inv_mapper

def recommend_movies_based_on_id(movie_id, k=5, metric='cosine', movie_df=None):
    """Recommend movies based on a given movie_id."""
    ratings = pd.read_csv('./data/user_ratings.csv')
    X, movie_mapper, movie_inv_mapper = create_X(ratings)
    
    X = X.T  # Transpose matrix
    neighbour_ids = []
    
    if movie_id not in movie_mapper:
        print(f"Movie ID {movie_id} not found in the dataset.")
        return []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)

    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    
    for i in range(1, k+1):  # Skip the first neighbour (itself)
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])

    # Map IDs back to titles
    if movie_df is not None:
        neighbour_titles = movie_df[movie_df['movie_id'].isin(neighbour_ids)]['title'].tolist()
        return neighbour_titles
    return neighbour_ids


recommend = recommend_movies_based_on_id(8009)
print(recommend)
    

