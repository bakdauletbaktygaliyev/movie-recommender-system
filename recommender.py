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

def recommend_movies_content_based(movie_name, user_id, top_n=5):
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


# Example usage
# movie_name = "Moana 2"  # Not used in this case for content-based filtering
# user_id = 1  # Example user ID
# recommendations = recommend_movies_content_based(movie_name, user_id)
# print(recommendations)
