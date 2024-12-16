import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import json

movies_path = "./data/movies.csv"
ratings_path = "./data/ratings_with_users.csv"

def load_data():
    # Load movies
    movies = pd.read_csv(movies_path)
    # Load ratings
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

def create_item_user_matrix(ratings_df, movies_df):
    """
    Create a movie-user rating matrix. 
    Rows = Movies, Columns = Users
    """
    # Pivot such that each row is a movie and each column is a user
    item_user = ratings_df.pivot_table(index='movie_id', columns='user_id', values='rating')
    # Fill NaN with 0 or some neutral value
    item_user = item_user.fillna(0)
    
    # It's a good idea to ensure that only movies present in both 
    # the ratings and movies dataframes are included.
    item_user = item_user[item_user.index.isin(movies_df['movie_id'])]
    
    return item_user

def train_item_knn_model(item_user_matrix, n_neighbors=5):
    """
    Train a kNN model on the movie-user matrix to find similar items.
    """
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    model_knn.fit(item_user_matrix)
    return model_knn

def get_movie_id_from_title(title, movies_df):
    """
    Given a movie title, return its movie_id.
    Assumes titles are unique. If not, further disambiguation is needed.
    """
    matches = movies_df[movies_df['title'].str.lower() == title.lower()]
    if len(matches) == 0:
        raise ValueError(f"No movie found with title '{title}'")
    return matches.iloc[0]['movie_id']

def recommend_similar_movies(title, movies_df, item_user_matrix, knn_model, top_n=5):
    """
    Recommend similar movies given a movie title.
    """
    # Convert title to movie_id
    movie_id = get_movie_id_from_title(title, movies_df)
    
    # If movie_id not in item_user_matrix, that means we have no data for it
    if movie_id not in item_user_matrix.index:
        raise ValueError(f"No rating data available for movie_id {movie_id}")
    
    # Get the movie vector (1 x num_users)
    movie_vector = item_user_matrix.loc[movie_id].values.reshape(1, -1)
    
    # Find neighbors
    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=top_n+1)
    # indices are the row indices of item_user_matrix; we need to map to movie_ids
    neighbor_ids = item_user_matrix.iloc[indices.flatten()[1:]].index
    
    # Retrieve movie titles for the neighbors
    similar_movies = movies_df[movies_df['movie_id'].isin(neighbor_ids)].copy()
    similar_movies['similarity'] = 1 - distances.flatten()[1:]  # cosine similarity = 1 - distance

    # Sort by similarity (descending)
    similar_movies = similar_movies.sort_values(by='similarity', ascending=False)
    return similar_movies[['movie_id', 'title', 'genres', 'release_year', 'similarity', 'rating']]

# Example Usage:
def item_based_collaborative_filtering(movie_title, top_n=5):
    # Load data
    movies_df, ratings_df = load_data()
    
    # Create item-user matrix
    item_user_matrix = create_item_user_matrix(ratings_df, movies_df)
    
    # Train item-based kNN model
    knn_model = train_item_knn_model(item_user_matrix, n_neighbors=5)
    
    return recommend_similar_movies(movie_title, movies_df, item_user_matrix, knn_model, top_n).to_json(orient='records', lines=False)


# -------------------------------------------------------------------------- #


def preprocess_data(movies):
    # Ensure 'genres' is a string and split it into a list
    movies['genres'] = movies['genres'].astype(str)
    movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|'))
    
    # Create one-hot encoding for genres
    all_genres = set([g for sublist in movies['genres_list'] for g in sublist])
    for g in all_genres:
        movies[f'genre_{g}'] = movies['genres_list'].apply(lambda x: 1 if g in x else 0)
    return movies

def get_user_profile(user_id, ratings, movies):
    # Get user ratings for movies present in our dataset
    user_ratings = ratings[ratings['user_id'] == user_id]
    merged = pd.merge(user_ratings, movies, on='movie_id', how='inner')
    
    merged.rename(columns={'rating_x': 'user_rating', 'rating_y': 'movie_rating'}, inplace=True)
    
    if merged.empty:
        # No movies rated by this user exist in our dataset
        return None
    
    genre_cols = [c for c in merged.columns if c.startswith('genre_')]
    # Weighted average of genres based on the user's ratings
    # weighted_genres = merged[genre_cols].T.dot(merged['rating'])
    weighted_genres = merged[genre_cols].T.dot(merged['user_rating'])
    
    # Normalize the user profile
    total = weighted_genres.sum()
    if total == 0:
        return None
    
    user_profile = weighted_genres / total
    return user_profile

def recommend_movies(user_id, top_n=10):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    movies = preprocess_data(movies)
    
    user_profile = get_user_profile(user_id, ratings, movies)
    
    # If we couldn't build a user profile, return empty JSON
    if user_profile is None:
        return json.dumps({"user_id": user_id, "recommendations": []})
    
    genre_cols = [c for c in movies.columns if c.startswith('genre_')]
    user_vector = user_profile.values.reshape(1, -1)
    movie_vectors = movies[genre_cols].values
    
    # Compute similarities
    similarities = cosine_similarity(user_vector, movie_vectors).flatten()
    movies['similarity'] = similarities
    
    # Exclude movies the user has already rated
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].unique()
    candidates = movies[~movies['movie_id'].isin(rated_movies)]
    
    # Sort by similarity
    recommendations = candidates.sort_values('similarity', ascending=False).head(top_n)
    
    # Convert recommendations to JSON including release_year and rating
    recommendations = recommendations[['movie_id', 'title', 'genres', 'release_year', 'rating', 'similarity']]
    return recommendations.to_json()
