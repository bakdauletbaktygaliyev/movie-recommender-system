import requests
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Milestone 1
# Define the base URL and API key for TMDb
API_KEY = 'API_KEY'  # Replace with your TMDb API key
BASE_URL = 'https://api.themoviedb.org/3'

# Function to fetch popular movies from TMDb API
def fetch_movies(page=1):
    """Fetch popular movies from TMDb API."""
    url = f"{BASE_URL}/movie/popular"
    params = {
        'api_key': API_KEY,
        'language': 'en-US',
        'page': page
    }
    response = requests.get(url, params=params)
    return response.json()

# List to store movie data
movies = []

# Fetch data from multiple pages (e.g., 1 to 50 pages)
for page in range(1, 51):  # Fetch data from the first two pages
    data = fetch_movies(page)
    for movie in data['results']:
        movies.append({
            'movie_id': movie['id'],  # Real movie ID
            'title': movie['title'],
            'genres': movie.get('genre_ids', []),  # Genre IDs
            'release_year': movie['release_date'].split('-')[0] if 'release_date' in movie else None,
            'rating': movie.get('vote_average', 0),
            'popularity': movie.get('popularity', 0),
        })


# Save data to a CSV file
df = pd.DataFrame(movies)
df.to_csv('./data/movies.csv', index=False)

# Print message
print("Data saved to movies.csv")

# Milestone 2
################################################################################################


# Load the dataset
df = pd.read_csv('./data/movies.csv')

# Display the first few rows to understand the structure of the dataset
print(df.head())

# Step 1: Handle Missing Values
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Option 1: Drop rows with missing values (if applicable)
# df.dropna(subset=['title', 'rating', 'release_year'], inplace=True)

# Option 2: Fill missing ratings with the average rating
df['rating'].fillna(df['rating'].mean(), inplace=True)

# Step 2: Normalize Numerical Data
# Normalize the 'rating' and 'popularity' columns using Min-Max Scaling
scaler = MinMaxScaler()
df[['rating', 'popularity']] = scaler.fit_transform(df[['rating', 'popularity']])

# Step 3: Encode Categorical Data (Genres)
# One-hot encode the 'genres' column
# First, we need to split the genres into individual genres
df['genres'] = df['genres'].apply(lambda x: x.strip('[]').replace(' ', '').split(','))

# One-hot encode the genres
genres_expanded = df['genres'].explode().unique()
for genre in genres_expanded:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

# Step 4: Remove Duplicates
# Remove duplicate rows based on the 'title' column
df.drop_duplicates(subset=['title'], inplace=True)

# Check the cleaned data
print("\nCleaned dataset:")
print(df.head())

# Step 5: Save Cleaned Data
df.to_csv('./data/cleaned_movies.csv', index=False)
print("\nCleaned data saved to cleaned_movies.csv")

# Milestone 3
################################################################################################


# Load the cleaned dataset
df = pd.read_csv('./data/cleaned_movies.csv')

# Display the first few rows to understand the dataset
print(df.head())

# Step 1: Statistical Summary
print("\nStatistical summary:")
print(df.describe())

# Step 2: Genre Distribution
# Sum each genre column to calculate total movies in each genre
genre_columns = [col for col in df.columns if col not in ['title', 'release_year', 'rating', 'popularity', 'genres']]
genre_counts = df[genre_columns].sum().sort_values(ascending=False)

# Plot genre distribution
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar', color='skyblue')
plt.title('Genre Distribution')
plt.xlabel('Genres')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 3: Rating Trends
# Distribution of ratings
plt.figure(figsize=(8, 5))
sns.histplot(df['rating'], bins=20, kde=True, color='purple')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Average rating by release year
avg_rating_by_year = df.groupby('release_year')['rating'].mean()

plt.figure(figsize=(12, 6))
avg_rating_by_year.plot(kind='line', marker='o', color='orange')
plt.title('Average Rating by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Correlation Analysis
# Heatmap for numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df[['rating', 'popularity']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Step 5: Movies Released Per Year
movies_per_year = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
movies_per_year.plot(kind='bar', color='green')
plt.title('Number of Movies Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# Additional Insight: Average Popularity by Genre
genre_popularity = {}
for genre in genre_columns:
    genre_popularity[genre] = df.loc[df[genre] == 1, 'popularity'].mean()

genre_popularity = pd.Series(genre_popularity).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
genre_popularity.plot(kind='bar', color='coral')
plt.title('Average Popularity by Genre')
plt.xlabel('Genres')
plt.ylabel('Average Popularity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Summarized Insights
print("\nSummarized Insights:")
print(f"Most popular genre: {genre_counts.idxmax()} ({genre_counts.max()} movies)")
print(f"Highest average rating: {df['rating'].max()}")
print(f"Year with most movie releases: {movies_per_year.idxmax()} ({movies_per_year.max()} movies)")


print("\n\n\n")
# Milestone 4
################################################################################################

# Collaborative Filtering
#############################################


# Load user ratings dataset
ratings = pd.read_csv('./data/user_ratings.csv')  # Columns: user_id, movie_id, rating

# Create user-item matrix
# user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Drop duplicate entries for the same user and movie
ratings = ratings.drop_duplicates(subset=['user_id', 'movie_id'])

# Now pivot the data
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)


# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)

# Function to recommend movies for a user
def recommend_movies_for_user(user_id, n_neighbors=5):
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector, n_neighbors=n_neighbors + 1)
    
    similar_users = indices.flatten()[1:]  # Exclude the user themselves
    recommended_movies = {}

    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.iloc[similar_user]
        for movie_id, rating in similar_user_ratings.items():
            if user_item_matrix.loc[user_id, movie_id] == 0 and rating > 0:  # Unrated by current user
                recommended_movies[movie_id] = recommended_movies.get(movie_id, 0) + rating

    # Sort movies by their weighted score
    sorted_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    return sorted_movies[:10]  # Return top 10 recommendations

# Example: Recommend movies for user with ID 1
recommended_movies = recommend_movies_for_user(user_id=1)
print("Recommended Movies:")
for movie_id, score in recommended_movies:
    print(f"Movie ID: {movie_id}, Score: {score}")
    

print("\n\n")
# Content-Based Filtering
#############################################

# Load movies dataset
movies = pd.read_csv('./data/cleaned_movies.csv')

# Combine relevant metadata into a single feature
movies['metadata'] = movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + movies['popularity'].astype(str)

# Convert metadata to feature vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
movie_features = vectorizer.fit_transform(movies['metadata'])

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(movie_features)

# Function to recommend movies based on a given movie
def recommend_movies_based_on_title(movie_title, n_neighbors=10):
    movie_index = movies[movies['title'] == movie_title].index[0]
    movie_vector = movie_features[movie_index]
    distances, indices = knn.kneighbors(movie_vector, n_neighbors=n_neighbors + 1)
    
    recommended_indices = indices.flatten()[1:]  # Exclude the movie itself
    recommended_movies = movies.iloc[recommended_indices]['title'].tolist()
    return recommended_movies

# Example: Recommend movies similar to "Inception"
recommended_movies = recommend_movies_based_on_title(movie_title="Inception")
print("Recommended Movies:")
for movie in recommended_movies:
    print(movie)
