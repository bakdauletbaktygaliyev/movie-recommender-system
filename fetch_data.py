import requests
import pandas as pd
import time

# Milestone 1
# Define the base URL and API key for TMDb
API_KEY = 'YOUR_API_KEY'  # Replace with your TMDb API key
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
# Example genre data you provided
genre_mapping = {
    28: 'Action',
    12: 'Adventure',
    16: 'Animation',
    35: 'Comedy',
    80: 'Crime',
    99: 'Documentary',
    18: 'Drama',
    10751: 'Family',
    14: 'Fantasy',
    36: 'History',
    27: 'Horror',
    10402: 'Music',
    9648: 'Mystery',
    10749: 'Romance',
    878: 'Science Fiction',
    10770: 'TV Movie',
    53: 'Thriller',
    10752: 'War',
    37: 'Western',
}


# Fetch data from multiple pages (e.g., 1 to 50 pages)
for page in range(401, 600):  # Fetch data from the first two pages
    data = fetch_movies(page)
    for movie in data['results']:
        genres_str = "|".join([genre_mapping.get(genre_id, "Unknown") for genre_id in movie.get('genre_ids', [])])
        movies.append({
            'movie_id': movie['id'],  # Real movie ID
            'title': movie['title'],
            'genres': genres_str,  # Genre IDs
            'release_year': movie['release_date'].split('-')[0] if 'release_date' in movie else None,
            'rating': movie.get('vote_average', 0),
            'popularity': movie.get('popularity', 0),
        })
        


# Save data to a CSV file
df = pd.DataFrame(movies)
df.to_csv('./data/movies.csv', index=False)

# Print message
print("Data saved to movies.csv")
print(len(movies))