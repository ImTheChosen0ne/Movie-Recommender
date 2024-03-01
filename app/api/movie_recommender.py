from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

movie_recommender = Blueprint('recommend', __name__)

def train_model(user_profile, movie_data, num_features=7, learning_rate=0.01, lambda_=0.1, epochs=300):
    rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}

    movie_features = []
    for index, movie in movie_data.iterrows():
        genres = ', '.join(movie['genres'])
        keywords = ', '.join(movie['keywords'])
        ratings = [rating_mapping.get(rating['rating'].lower(), 0) for rating in movie['ratings']]
        rating_avg = sum(ratings) / len(ratings) if ratings else 0

        movie_length = movie['runtime']
        director = movie['director']
        actors = ', '.join(movie['casts'])
        release_year = movie['releaseDate']
        companies = ', '.join(movie['companies'])

        movie_features.append([genres, keywords, rating_avg, director, actors, release_year, companies, movie_length])

    user_preferences = []
    for rating in user_profile["profileRatings"]:
        rating_numeric = rating_mapping.get(rating["rating"].lower())
        if rating_numeric is not None:
            user_preferences.append({"movieId": rating["movie"]["movieId"], "rating": rating_numeric})

    # viewed_movies = []
    # for viewed_movie in user_profile["viewedMovies"]:
    #     if not any(pref["movieId"] == viewed_movie["movie"]["movieId"] for pref in user_preferences):
    #         viewed_rating_numeric = rating_mapping.get(viewed_movie["ratings"][0]["rating"].lower(), 0)
    #         viewed_movies.append({"movieId": viewed_movie["movie"]["movieId"], "rating": viewed_rating_numeric})

    # user_preferences.extend(viewed_movies)

    movie_features_array = np.array(movie_features)
    user_preferences_array = np.array(user_preferences)

    def cofi_cost_func_v(X, W, b, Y, R, lambda_):
        j = tf.linalg.matmul(X, tf.transpose(W)) + b - Y
        J = 0.5 * tf.reduce_sum((j * R) ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
        return J

    num_users_r = 1
    num_movies_r = len(movie_data)

    Y_r = np.zeros((num_movies_r, num_users_r))
    for rating in user_preferences:
        movie_index = movie_data.index[movie_data['movieId'] == rating['movieId']][0]
        Y_r[movie_index, 0] = rating['rating']

    R_r = np.ones((num_movies_r, num_users_r))

    X_r = tf.Variable(np.random.rand(num_movies_r, num_features), dtype=tf.float64)
    W_r = tf.Variable(np.random.rand(num_users_r, num_features), dtype=tf.float64)
    b_r = tf.Variable(np.random.rand(1, num_users_r), dtype=tf.float64)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            cost = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, lambda_)

        gradients = tape.gradient(cost, [X_r, W_r, b_r])
        X_r.assign_sub(learning_rate * gradients[0])
        W_r.assign_sub(learning_rate * gradients[1])
        b_r.assign_sub(learning_rate * gradients[2])

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}, Cost: {cost.numpy():.4f}")

    return X_r, W_r, b_r

@movie_recommender.route('/user_recommendations', methods=['POST'])
def get_recommendations():
    try:
        profile_data = request.json.get('profile')
        movie_data = request.json.get('movies')
        movie_df = pd.DataFrame(movie_data)

        required_fields = ['profileId', 'profileRatings']
        for field in required_fields:
            if field not in profile_data:
                # print({'error': f'Missing required field: {field}'})

                return jsonify({'error': f'Missing required field: {field}'}), 400

        X, W, b = train_model(profile_data, movie_df)

        predicted_ratings = tf.matmul(X, tf.transpose(W)) + b
        top_movie_indices = tf.argsort(predicted_ratings, axis=0, direction='DESCENDING')[:25]
        recommended_movies = movie_df.iloc[top_movie_indices.numpy().flatten()].to_dict(orient='records')

        return jsonify({'recommendations': recommended_movies}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


def find_similar_movies(movie_name, movie_data, num_similar_movies=25):
    movie_index = movie_data.index[movie_data['title'] == movie_name["title"]].tolist()[0]

    rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}

    def extract_ratings(rating_list):
        numerical_ratings = [rating_mapping[rating["rating"].lower()] for rating in rating_list]
        return sum(numerical_ratings) / len(numerical_ratings) if len(numerical_ratings) > 0 else 0

    movie_data['ratings'] = movie_data['ratings'].apply(extract_ratings)

    movie_features = [', '.join(movie_data.iloc[i]['genres']) + ', ' + ', '.join(movie_data.iloc[i]['keywords']) + ', ' + str(movie_data.iloc[i]['ratings']) for i in range(len(movie_data))]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(movie_features)

    cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    similar_movie_indices = cosine_similarities.argsort()[-num_similar_movies-1:-1][::-1]

    similar_movies = movie_data.iloc[similar_movie_indices].to_dict(orient='records')

    return similar_movies

@movie_recommender.route('/similar_movies', methods=['POST'])
def get_similar_movies():
    try:
        movie_name = request.json.get('movie_name')

        if not movie_name:
            return jsonify({'error': 'Movie name is required.'}), 400

        movie_data = request.json.get('movies')
        movie_df = pd.DataFrame(movie_data)

        similar_movies = find_similar_movies(movie_name, movie_df)

        return jsonify({'similar_movies': similar_movies}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    movie_recommender.run(debug=True)
