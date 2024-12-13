import pandas as pd
from sklearn.neighbors import NearestNeighbors

def recommend_movies_collaborative_knn(movie_name, user_id, top_n=5):
    # Load the ratings and movies data
    ratings_df = pd.read_csv("./data/ratings_with_users.csv")
    movies_df = pd.read_csv("./data/movies.csv")
    
    # Merge ratings with movie titles
    movie_ratings = pd.merge(
        ratings_df,
        movies_df[['movie_id', 'title', 'genres', 'rating']],
        on='movie_id',
        how='inner'
    )
    
    # Use 'rating_x' as the user-specific rating
    movie_ratings.rename(columns={'rating_x': 'rating'}, inplace=True)
    
    # Create a user-item matrix (user_id as rows, movie titles as columns)
    user_item_matrix = movie_ratings.pivot_table(index='user_id', columns='title', values='rating', fill_value=0)

    # Check if the movie_name exists in the dataset
    if movie_name not in user_item_matrix.columns:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")

    # Apply KNN to find similar users based on movie ratings
    knn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine')  # +1 to include the target user itself
    knn.fit(user_item_matrix.values)  # Pass only values to avoid feature name issue
    
    # Get the user index for the target user
    user_idx = user_item_matrix.index.get_loc(user_id)
    
    # Get distances and indices of similar users
    distances, indices = knn.kneighbors(user_item_matrix.iloc[user_idx].values.reshape(1, -1))

    # Extract similar user indices and their similarity scores
    similar_user_ids = user_item_matrix.index[indices.flatten()[1:]]
    similar_user_distances = distances.flatten()[1:]

    # Find movies that similar users liked, which the target user hasn't rated
    similar_user_ratings = movie_ratings[movie_ratings['user_id'].isin(similar_user_ids)]
    recommended_movies = similar_user_ratings[~similar_user_ratings['title'].isin(
        movie_ratings[movie_ratings['user_id'] == user_id]['title']
    )]

    # Calculate the average rating of movies liked by similar users
    movie_recommendations = recommended_movies.groupby('title').agg(
        Similarity=('rating', 'mean'),
        Genre=('genres', 'first'),
        Rating=('rating', 'mean')
    ).reset_index()

    # Sort recommendations based on average rating
    movie_recommendations = movie_recommendations.sort_values(by='Rating', ascending=False).head(top_n)

    # Convert to JSON format
    recommendations_json = movie_recommendations.to_json(orient='records', lines=False)
    
    return recommendations_json


# Example usage
# movie_name = "Moana 2"
# user_id = 1
# recommendations = recommend_movies_collaborative_knn(movie_name, user_id)
# print(recommendations)

import pandas as pd

def recommend_movies_content_based(user_id, top_n=5):
    # Load the ratings and movies data
    ratings_df = pd.read_csv("./data/ratings_with_users.csv")
    movies_df = pd.read_csv("./data/movies.csv")

    # Merge ratings with movie titles
    movie_ratings = pd.merge(
        ratings_df,
        movies_df[['movie_id', 'title', 'genres', 'rating']],
        on='movie_id',
        how='inner'
    )

    # Use 'rating_x' as the user-specific rating
    movie_ratings.rename(columns={'rating_x': 'rating'}, inplace=True)

    # Fetch high-rated movies for the user (e.g., rating >= 4)
    high_rated_movies = movie_ratings[(movie_ratings['user_id'] == user_id) & (movie_ratings['rating'] >= 4)]

    # Get the genres of those high-rated movies
    high_rated_genres = high_rated_movies['genres'].str.split('|').explode().unique()

    # Ensure that genres are valid (non-null and strings)
    movies_df['genres'] = movies_df['genres'].fillna('')  # Fill missing genres with an empty string

    # Find movies with similar genres to the high-rated movies
    similar_movies = movies_df[movies_df['genres'].apply(
        lambda x: any(genre in x.split('|') for genre in high_rated_genres) if isinstance(x, str) else False
    )]

    # Filter out movies the user has already rated
    recommended_movies = similar_movies[~similar_movies['title'].isin(high_rated_movies['title'])]

    # Sort recommendations based on the rating
    recommended_movies = recommended_movies.sort_values(by='rating', ascending=False).head(top_n)

    # Convert to JSON format
    recommendations_json = recommended_movies[['title', 'genres', 'rating']].to_json(orient='records', lines=False)
    
    return recommendations_json


# # Example usage
# user_id = 1  # Example user ID
# recommendations = recommend_movies_content_based(user_id)
# print(recommendations)



def recommend_movies_content_based(user_id, top_n=5):
    print(f"Starting recommendation process for user_id: {user_id}")

    # Load the ratings and movies data
    print("Loading ratings and movies datasets...")
    ratings_df = pd.read_csv("./data/ratings_with_users.csv")
    movies_df = pd.read_csv("./data/movies.csv")
    print(f"Ratings dataset shape: {ratings_df.shape}")
    print(f"Movies dataset shape: {movies_df.shape}")

    # Merge ratings with movie titles
    print("Merging ratings with movie titles...")
    movie_ratings = pd.merge(
        ratings_df,
        movies_df[['movie_id', 'title', 'genres', 'rating']],
        on='movie_id',
        how='inner'
    )
    movie_ratings.rename(columns={'rating_x': 'rating'}, inplace=True)
    print(f"Merged dataset shape: {movie_ratings.shape}")

    # Fetch high-rated movies for the user (e.g., rating >= 4)
    print(f"Filtering high-rated movies for user_id: {user_id}")
    high_rated_movies = movie_ratings[(movie_ratings['user_id'] == user_id) & (movie_ratings['rating'] >= 6)]
    print(f"High-rated movies found: {high_rated_movies.shape[0]}")
    print(f"High-rated movies titles: {high_rated_movies['title'].tolist()}")

    # Get the genres of those high-rated movies
    high_rated_genres = high_rated_movies['genres'].str.split('|').explode().unique()
    print(f"Genres extracted from high-rated movies: {high_rated_genres}")

    # Ensure that genres are valid (non-null and strings)
    print("Filling missing genres in movies dataset...")
    movies_df['genres'] = movies_df['genres'].fillna('')

    # Find movies with similar genres to the high-rated movies
    print("Finding movies with similar genres...")
    similar_movies = movies_df[movies_df['genres'].apply(
        lambda x: any(genre in x.split('|') for genre in high_rated_genres) if isinstance(x, str) else False
    )]
    print(f"Similar movies found: {similar_movies.shape[0]}")

    # Filter out movies the user has already rated
    print("Filtering out movies already rated by the user...")
    recommended_movies = similar_movies[~similar_movies['title'].isin(high_rated_movies['title'])]
    print(f"Movies left after filtering: {recommended_movies.shape[0]}")

    # Sort recommendations based on the rating
    print("Sorting recommendations by rating...")
    recommended_movies = recommended_movies.sort_values(by='rating', ascending=False).head(top_n)
    print(f"Top {top_n} recommended movies:")
    print(recommended_movies[['title', 'genres', 'rating']])

    # Convert to JSON format
    recommendations_json = recommended_movies[['title', 'genres', 'rating']].to_json(orient='records', lines=False)
    print("Recommendation process completed. Returning JSON data.")
    
    return recommendations_json


# Example usage
# user_id = 1  # Example user ID
# recommendations = recommend_movies_content_based(user_id)
# print(recommendations)

from sklearn.neighbors import NearestNeighbors
import pandas as pd

def recommend_movies_collaborative_knn(movie_name, user_id, top_n=5):
    print(f"Starting collaborative filtering recommendation for user_id: {user_id} based on movie: '{movie_name}'")

    # Load the ratings and movies data
    print("Loading datasets...")
    ratings_df = pd.read_csv("./data/ratings_with_users.csv")
    movies_df = pd.read_csv("./data/movies.csv")
    print(f"Ratings dataset shape: {ratings_df.shape}")
    print(f"Movies dataset shape: {movies_df.shape}")

    # Merge ratings with movie titles
    print("Merging ratings with movie titles...")
    movie_ratings = pd.merge(
        ratings_df,
        movies_df[['movie_id', 'title', 'genres', 'rating']],
        on='movie_id',
        how='inner'
    )
    movie_ratings.rename(columns={'rating_x': 'rating'}, inplace=True)
    print(f"Merged dataset shape: {movie_ratings.shape}")

    # Create a user-item matrix
    print("Creating user-item matrix...")
    user_item_matrix = movie_ratings.pivot_table(index='user_id', columns='title', values='rating', fill_value=0)
    print(f"User-item matrix shape: {user_item_matrix.shape}")

    # Check if the movie_name exists in the dataset
    if movie_name not in user_item_matrix.columns:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")

    # Initialize KNN model
    print("Initializing KNN model...")
    knn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine')  # +1 to include the target user itself
    knn.fit(user_item_matrix.values)  # Pass only values to avoid feature name issues
    
    # Get the user index for the target user
    print(f"Finding similar users for user_id: {user_id}...")
    try:
        user_idx = user_item_matrix.index.get_loc(user_id)
    except KeyError:
        raise ValueError(f"User ID '{user_id}' not found in the dataset.")
    
    # Compute distances and indices for similar users
    distances, indices = knn.kneighbors(user_item_matrix.iloc[user_idx].values.reshape(1, -1))
    print(f"Similar users found: {indices.flatten()[1:].tolist()} (excluding target user)")
    print(f"Distances to similar users: {distances.flatten()[1:].tolist()}")

    # Extract similar user IDs and their distances
    similar_user_ids = user_item_matrix.index[indices.flatten()[1:]]
    similar_user_distances = distances.flatten()[1:]

    # Find movies that similar users liked and the target user hasn't rated
    print("Finding movies liked by similar users but not rated by target user...")
    similar_user_ratings = movie_ratings[movie_ratings['user_id'].isin(similar_user_ids)]
    recommended_movies = similar_user_ratings[~similar_user_ratings['title'].isin(
        movie_ratings[movie_ratings['user_id'] == user_id]['title']
    )]
    print(f"Movies found for recommendation: {recommended_movies.shape[0]}")

    # Calculate the average rating and genre for recommended movies
    print("Aggregating recommendations...")
    movie_recommendations = recommended_movies.groupby('title').agg(
        Similarity=('rating', 'mean'),
        Genre=('genres', 'first'),
        Rating=('rating', 'mean')
    ).reset_index()

    # Sort recommendations by rating
    print("Sorting recommendations by average rating...")
    movie_recommendations = movie_recommendations.sort_values(by='Rating', ascending=False).head(top_n)
    print(f"Top {top_n} recommendations:")
    print(movie_recommendations[['title', 'Genre', 'Rating']])

    # Convert to JSON format
    print("Converting recommendations to JSON format...")
    recommendations_json = movie_recommendations.to_json(orient='records', lines=False)
    print("Recommendation process completed.")

    return recommendations_json

# Example usage
movie_name = "Moana 2"
user_id = 1
recommendations = recommend_movies_collaborative_knn(movie_name, user_id)
print(recommendations)