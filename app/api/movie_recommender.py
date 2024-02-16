from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

movie_recommender = Blueprint('recommend', __name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

movies_csv_path = os.path.join(current_dir, "movies.csv")

movie_data = pd.read_csv(movies_csv_path, sep=";")

def train_model(user_profile, movie_data, num_features=3, learning_rate=0.01, lambda_=0.1, epochs=200):
    # movie_features = []
    # for index, movie in movie_data.iterrows():
    #     genres = movie["genres"]
    #     keywords = movie["keywords"]
    #     movie_features.append([genres, keywords])

    # user_preferences = []
    # for rating in user_profile["ratings"]:
    #     rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}
    #     rating_numeric = rating_mapping.get(rating["rating"].lower())
    #     if rating_numeric is not None:
    #         user_preferences.append({"movieId": rating["movieId"], "rating": rating_numeric})

    # movie_features_array = np.array(movie_features)
    # user_preferences_array = np.array(user_preferences)

    # def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    #     j = tf.linalg.matmul(X, tf.transpose(W)) + b - Y
    #     J = 0.5 * tf.reduce_sum((j * R) ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
    #     return J

    # num_users_r = 1
    # num_movies_r = len(movie_data)

    # Y_r = np.zeros((num_movies_r, num_users_r))
    # for rating in user_preferences:
    #     movie_index = movie_data.index[movie_data['movieId'] == rating['movieId']][0]
    #     Y_r[movie_index, 0] = rating['rating']

    # R_r = np.ones((num_movies_r, num_users_r))

    # X_r = tf.Variable(np.random.rand(num_movies_r, num_features), dtype=tf.float32)
    # W_r = tf.Variable(np.random.rand(num_users_r, num_features), dtype=tf.float32)
    # b_r = tf.Variable(np.random.rand(1, num_users_r), dtype=tf.float32)

    # for epoch in range(epochs):
    #     with tf.GradientTape() as tape:
    #         cost = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, lambda_)

    #     gradients = tape.gradient(cost, [X_r, W_r, b_r])
    #     X_r.assign_sub(learning_rate * gradients[0])
    #     W_r.assign_sub(learning_rate * gradients[1])
    #     b_r.assign_sub(learning_rate * gradients[2])

    #     if epoch % 10 == 0:
    #         print(f"Epoch {epoch + 1}, Cost: {cost.numpy():.4f}")

    movie_features = []
    for index, movie in movie_data.iterrows():
        genres = movie["genres"]
        keywords = movie["keywords"]
        rating = movie["ratings"]

        movie_features.append([genres, keywords, rating])

    movie_features_array = np.array(movie_features)

    user_preferences = []
    for rating in user_profile["ratings"]:
        rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}
        rating_numeric = rating_mapping.get(rating["rating"].lower())

        if rating_numeric is not None:
            user_preferences.append({"movieId": rating["movieId"], "rating": rating_numeric})

    movie_features_array = np.array(movie_features)
    user_preferences_array = np.array(user_preferences)

    def cofi_cost_func_v(X, W, b, Y, R, lambda_):
        """
        Returns the cost for collaborative filtering
        Vectorized for speed using TensorFlow operations.
        """
        j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
        J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
        return J

    num_users_r = 1
    num_movies_r = len(movie_data)
    num_features_r = 3

    # Prepare user ratings
    # num_movies = len(movie_data)
    # num_users = 1  # Only one user in this case

    Y_r = np.zeros((num_movies_r, num_movies_r))
    for rating in user_preferences:
        movie_index = movie_data.index[movie_data['movieId'] == rating['movieId']][0]
        Y_r[movie_index, 0] = rating['rating']

    R_r = np.ones((num_movies_r, num_users_r))

    for movie in user_profile["viewedMovies"]:
        movie_index = movie_data.index[movie_data['movieId'] == movie['movieId']][0]
        Y_r[movie_index, 0] = 0

    X_r = tf.Variable(np.random.rand(num_movies_r, num_features_r), dtype=tf.float32)
    W_r = tf.Variable(np.random.rand(num_users_r, num_features_r), dtype=tf.float32)
    b_r = tf.Variable(np.random.rand(1, num_users_r), dtype=tf.float32)

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
        user_data = request.json

        required_fields = ['userId', 'ratings']
        for field in required_fields:
            if field not in user_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        movie_data = pd.read_csv("movies.csv", sep=";")

        X, W, b = train_model(user_data, movie_data)

        predicted_ratings = tf.matmul(X, tf.transpose(W)) + b
        top_movie_indices = tf.argsort(predicted_ratings, axis=0, direction='DESCENDING')[:25]
        recommended_movies = movie_data.iloc[top_movie_indices.numpy().flatten()]["title"].values

        return jsonify({'recommendations': recommended_movies}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def find_similar_movies(movie_name, num_similar_movies=25):

    movie_index = movie_data.index[movie_data['title'] == movie_name["title"]].tolist()[0]

    # movie_features = movie_data['genres'] + ' ' + movie_data['keywords']
    # tfidf_vectorizer = TfidfVectorizer()
    # tfidf_matrix = tfidf_vectorizer.fit_transform(movie_features)

    # cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    # similar_movie_indices = cosine_similarities.argsort()[-num_similar_movies-1:-1][::-1]

    # similar_movie_titles = movie_data.iloc[similar_movie_indices]['title'].values

    movie_features = movie_data['genres'] + ' ' + movie_data['keywords'] + ' ' + movie_data['ratings']

    rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}

    def extract_ratings(rating_string):
        ratings = rating_string.split('|')
        numerical_ratings = [rating_mapping[r.split(',')[1]] for r in ratings]
        return sum(numerical_ratings) / len(numerical_ratings) if len(numerical_ratings) > 0 else 0

    movie_data['ratings'] = movie_data['ratings'].apply(extract_ratings)

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(movie_features)

    cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    similar_movie_indices = cosine_similarities.argsort()[-num_similar_movies-1:-1][::-1]

    similar_movie_titles = movie_data.iloc[similar_movie_indices]['title'].values

    return similar_movie_titles

@movie_recommender.route('/similar_movies', methods=['POST'])
def get_similar_movies():
    try:
        movie_name = request.json.get('movie_name')

        if not movie_name:
            return jsonify({'error': 'Movie name is required.'}), 400

        similar_movies = find_similar_movies(movie_name)

        similar_movies_list = similar_movies.tolist()

        return jsonify({'similar_movies': similar_movies_list}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    movie_recommender.run(debug=True)
