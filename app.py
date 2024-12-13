from flask import Flask, request, jsonify
from flask_cors import CORS
import knn as knn
import recommender as recommender

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Movie Recommendation System!"

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('title')
    try:
        # recommendations = knn.recommend_movies_by_genre(movie_title)
        # return jsonify(recommendations)
        # return jsonify(recommendations.to_dict(orient='records'))
        recommendations = recommender.recommend_movies_collaborative_knn()
    except IndexError:
        return jsonify({"error": "Movie not found!"}), 404

if __name__ == '__main__':
    app.run(debug=True)
