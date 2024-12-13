import requests
import json
import csv

API_KEY = '6901af3c05b06d80e298462dfd88e8bf'  # Replace with your TMDb API key
BASE_URL = 'https://api.themoviedb.org/3'

url = f"{BASE_URL}/genre/movie/list"

params = {
    'api_key': API_KEY,
    'language': 'en-US',
}
response = requests.get(url, params=params)
json_data = response.json()

print(json_data)