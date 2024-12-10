import requests
import pandas as pd
import numpy as np
import random
import string

# Define helper functions
def generate_username(user_id):
    """Generate a username based on user ID."""
    return f"user{user_id}"

def generate_password(length=8):
    """Generate a random password."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Load movies_with_ids.csv
movies = pd.read_csv('./data/cleaned_movies.csv')

# Use real movie IDs from the movies dataset
movie_ids = movies['movie_id'].values

# Simulate user IDs
user_ids = np.arange(1, 101)  # 100 unique users

# Generate synthetic user ratings
data = {
    'user_id': np.random.choice(user_ids, size=500, replace=True),
    'movie_id': np.random.choice(movie_ids, size=500, replace=True),  # Real movie IDs
    'rating': np.random.randint(1, 6, size=500),  # Ratings from 1 to 5
    'username': [generate_username(uid) for uid in np.random.choice(user_ids, size=500, replace=True)],
    'password': [generate_password() for _ in range(500)]
}

# Create a DataFrame and save to CSV
ratings = pd.DataFrame(data)
ratings.to_csv('./data/user_ratings.csv', index=False)
print("User ratings with real movie IDs saved!")
