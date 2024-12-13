import pandas as pd
from sklearn.neighbors import NearestNeighbors

def find_similar_movies_knn(movie_name, top_n=5):
    movies_df = pd.read_csv('./data/movies.csv')  # Load movie data with title and genre columns
    ratings_df = pd.read_csv('./data/ratings_with_users.csv')  # Load ratings data with movie_id, user_id, and rating
    
    # Merge movies with ratings to get the movie titles alongside their ratings
    movie_ratings = pd.merge(ratings_df, movies_df[['movie_id', 'title']], on='movie_id')
    
    # Create a user-item matrix (rows are users, columns are movies, values are ratings)
    user_item_matrix = movie_ratings.pivot_table(index='user_id', columns='title', values='rating', fill_value=0)
    
    # Check if the movie_name exists in the dataset
    if movie_name not in user_item_matrix.columns:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")
    
    # Find the index of the movie in the user-item matrix
    movie_idx = user_item_matrix.columns.get_loc(movie_name)
    
    # Apply KNN to find similar movies
    knn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine')  # +1 because the movie itself is also considered similar
    knn.fit(user_item_matrix.T)  # Transpose to treat movies as rows
    
    # Find similar movies
    distances, indices = knn.kneighbors(user_item_matrix.T.iloc[movie_idx].values.reshape(1, -1))

    # Get the similar movie titles and their corresponding distances
    similar_movie_titles = user_item_matrix.columns[indices.flatten()[1:]]
    similar_movie_distances = distances.flatten()[1:]
    

    # Create a DataFrame for better display of the results
    similar_movies = pd.DataFrame({
        'Movie': similar_movie_titles,
        'Similarity': 1 - similar_movie_distances  # Cosine similarity: 1 - distance
    })
    
    return similar_movies

# # Example usage
# movies_df = pd.read_csv('./data/movies.csv')  # Load your movies data
# ratings_df = pd.read_csv('./data/ratings_with_users.csv')  # Load your ratings data

# # Find similar movies to "Venom: The Last Dance"
# similar_movies = find_similar_movies_knn("Deepwater Horizon", movies_df, ratings_df, top_n=10)
# print(similar_movies)


def recommend_movies_for_user(user_id, movies_df, ratings_df, top_n=5):
    # Merge ratings with movie titles
    user_ratings = pd.merge(ratings_df, movies_df[['movie_id', 'title']], on='movie_id')

    # Create a user-item matrix (rows are users, columns are movies, values are ratings)
    user_item_matrix = user_ratings.pivot_table(index='user_id', columns='title', values='rating', fill_value=0)

    # Check if the user exists in the dataset
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User with ID {user_id} not found in the dataset.")

    # Get the ratings for the specified user
    user_ratings_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)

    # Apply KNN to find similar users
    knn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine')  # +1 to exclude the user itself
    knn.fit(user_item_matrix.values)  # Fit on the entire user-item matrix

    # Find similar users
    distances, indices = knn.kneighbors(user_ratings_vector)

    # Get the indices of the similar users (excluding the user itself)
    similar_user_indices = indices.flatten()[1:]

    # Gather the ratings from similar users
    similar_user_ratings = user_item_matrix.iloc[similar_user_indices]

    # Find movies rated highly by similar users that the current user hasn't rated yet
    recommended_movies = {}
    for idx in similar_user_ratings.index:
        for movie, rating in similar_user_ratings.loc[idx].items():
            if rating >= 7 and user_item_matrix.loc[user_id, movie] == 0:  # Only recommend movies that the user hasn't rated and have high ratings
                if movie not in recommended_movies:
                    recommended_movies[movie] = rating
    
    # Sort recommended movies by rating (descending order)
    recommended_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)

    # Return the top N recommended movies
    top_recommended_movies = recommended_movies[:top_n]
    top_movie_titles = [movie for movie, _ in top_recommended_movies]

    return top_movie_titles

# # Example usage
# movies_df = pd.read_csv('./data/movies.csv')  # Load your movies data
# ratings_df = pd.read_csv('./data/ratings_with_users.csv')  # Load your ratings data

# # Recommend movies for user with ID 1
# recommended_movies = recommend_movies_for_user(user_id=1, movies_df=movies_df, ratings_df=ratings_df, top_n=5)
# print("Recommended Movies for User 1:")
# for movie in recommended_movies:
#     print(movie)


def recommend_movies_by_genre(movie_name, top_n=5, use_popularity=True):
    movies_df = pd.read_csv('./data/movies.csv')  # Load movie data with title and genre columns
    ratings_df = pd.read_csv('./data/ratings_with_users.csv')  # Load ratings data with movie_id, user_id, and rating
    
    # Merge ratings with movie titles and genres
    movie_data = pd.merge(ratings_df, movies_df[['movie_id', 'title', 'genres']], on='movie_id')
    
    # Check if the given movie exists in the dataset
    if movie_name not in movie_data['title'].values:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")
    
    # Get the genre(s) of the specified movie
    movie_genres = movies_df[movies_df['title'] == movie_name]['genres'].values[0]
    
    # Filter movies based on the same genre(s)
    genre_movies = movie_data[movie_data['genres'] == movie_genres]
    
    # Optionally sort by popularity (average rating or number of ratings)
    if use_popularity:
        # Calculate average rating per movie
        popularity = genre_movies.groupby('title')['rating'].mean().reset_index()
        # Sort by average rating (descending)
        sorted_movies = popularity.sort_values(by='rating', ascending=False)
    else:
        # Sort by the number of ratings (ascending)
        popularity = genre_movies.groupby('title')['rating'].count().reset_index()
        sorted_movies = popularity.sort_values(by='rating', ascending=False)
    
    # Exclude the original movie from the recommendations
    sorted_movies = sorted_movies[sorted_movies['title'] != movie_name]
    
    # Get the top N recommended movies
    top_recommended_movies = sorted_movies.head(top_n)['title'].values
    
    return top_recommended_movies

# Example usage


# Recommend movies based on genre and popularity for the movie "The Matrix"
# recommended_movies = recommend_movies_by_genre(movie_name="The Matrix", top_n=5)
# print("Recommended Movies based on Genre and Popularity:")
# for movie in recommended_movies:
#     print(movie)

def recommend_movies_by_genre(movie_name, top_n=5, use_popularity=True):
    # Load data
    movies_df = pd.read_csv('./data/movies.csv')  # Load movie data with title, genres, release_year
    ratings_df = pd.read_csv('./data/ratings_with_users.csv')  # Load ratings data with movie_id, user_id, rating

    # Merge ratings with movie titles and genres
    movie_data = pd.merge(ratings_df, movies_df[['movie_id', 'title', 'genres', 'release_year']], on='movie_id')

    # Check if the given movie exists in the dataset
    if movie_name not in movie_data['title'].values:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")

    # Get the genre(s) of the specified movie
    movie_genres = movies_df[movies_df['title'] == movie_name]['genres'].values[0]

    # Filter movies based on the same genre(s)
    genre_movies = movie_data[movie_data['genres'] == movie_genres]

    # Optionally sort by popularity (average rating or number of ratings)
    if use_popularity:
        # Calculate average rating per movie
        popularity = genre_movies.groupby(['title', 'release_year'])['rating'].mean().reset_index()
        # Sort by average rating (descending)
        sorted_movies = popularity.sort_values(by='rating', ascending=False)
    else:
        # Sort by the number of ratings
        popularity = genre_movies.groupby(['title', 'release_year'])['rating'].count().reset_index()
        sorted_movies = popularity.sort_values(by='rating', ascending=False)

    # Exclude the original movie from the recommendations
    sorted_movies = sorted_movies[sorted_movies['title'] != movie_name]

    # Get the top N recommended movies
    recommendations = sorted_movies.head(top_n)

    # Return the required columns
    return recommendations[['title', 'rating', 'release_year']]

