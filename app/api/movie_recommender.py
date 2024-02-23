from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import requests

movie_recommender = Blueprint('recommend', __name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

movies_csv_path = os.path.join(current_dir, "movies.csv")

# movie_data = pd.read_csv(movies_csv_path, sep=";")

# def fetch_movie_data_from_api(jwt_token):
#     api_url = "http://localhost:8080/api/movies/"
#     headers = {"Authorization": f"Bearer {jwt_token}"}

#     try:
#         response = requests.get(api_url, headers=headers)

#         if response.status_code == 200:
#             movie_data = response.json()
#             return movie_data
#         else:
#             print(f"Error: {response.status_code} - {response.text}")
#             return None
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# @movie_recommender.route('/api/movies', methods=['GET'])
# def get_movies(jwt_token):
#     spring_boot_url = "http://localhost:8080/api/movies"  # Assuming Spring Boot backend runs on port 8080
#     headers = {"Content-Type": "application/json", "Authorization": f"Bearer {jwt_token}"}

#     try:
#         response = requests.get(spring_boot_url, headers=headers)
#         if response.status_code == 200:
#             return jsonify(response.json())
#         else:
#             return jsonify({"error": "Failed to fetch movie data"}), response.status_code
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

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

    # movie_features = []
    # for index, movie in movie_data.iterrows():
    #     genres = movie["genres"]
    #     keywords = movie["keywords"]
    #     rating = movie["ratings"]

    #     movie_features.append([genres, keywords, rating])

    # movie_features_array = np.array(movie_features)

    # user_preferences = []
    # for rating in user_profile["ratings"]:
    #     rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}
    #     rating_numeric = rating_mapping.get(rating["rating"].lower())

    #     if rating_numeric is not None:
    #         user_preferences.append({"movieId": rating["movieId"], "rating": rating_numeric})

    # movie_features_array = np.array(movie_features)
    # user_preferences_array = np.array(user_preferences)

    # def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    #     """
    #     Returns the cost for collaborative filtering
    #     Vectorized for speed using TensorFlow operations.
    #     """
    #     j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    #     J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    #     return J

    # num_users_r = 1
    # num_movies_r = len(movie_data)
    # num_features_r = 3

    # # Prepare user ratings
    # # num_movies = len(movie_data)
    # # num_users = 1  # Only one user in this case

    # Y_r = np.zeros((num_movies_r, num_movies_r))
    # for rating in user_preferences:
    #     movie_index = movie_data.index[movie_data['movieId'] == rating['movieId']][0]
    #     Y_r[movie_index, 0] = rating['rating']

    # R_r = np.ones((num_movies_r, num_users_r))

    # for movie in user_profile["viewedMovies"]:
    #     movie_index = movie_data.index[movie_data['movieId'] == movie['movieId']][0]
    #     Y_r[movie_index, 0] = 0

    # X_r = tf.Variable(np.random.rand(num_movies_r, num_features_r), dtype=tf.float32)
    # W_r = tf.Variable(np.random.rand(num_users_r, num_features_r), dtype=tf.float32)
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

    # return X_r, W_r, b_r

        # Convert movie data to feature vectors
    movie_features = []
    for movie in movie_data:
        genres = ', '.join(movie['genres'])
        keywords = ', '.join(movie['keywords'])
        rating = ', '.join([rating['rating'] for rating in movie['ratings']])

        movie_features.append([genres, keywords, rating])

    movie_features_array = np.array(movie_features)
    print(movie_features_array)
    # Convert user ratings to appropriate format
    user_preferences = []
    for rating in user_profile["profileRatings"]:
        rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}
        rating_numeric = rating_mapping.get(rating["rating"].lower())

        if rating_numeric is not None:
            user_preferences.append({"movieId": rating["movie"]["movieId"], "rating": rating_numeric})


    user_preferences_array = np.array(user_preferences)


    # Define cost function
    def cofi_cost_func_v(X, W, b, Y, R, lambda_):
        j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
        J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
        return J

    num_users_r = 1
    num_movies_r = len(movie_data)
    num_features_r = num_features

    # Prepare user ratings
    Y_r = np.zeros((num_movies_r, num_users_r))
    for movie_id, rating in user_preferences_array:
        movie_index = movie_data.index(movie_id)
        Y_r[movie_index, 0] = rating

    R_r = np.ones((num_movies_r, num_users_r))

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
        # Retrieve JWT token from request headers
        # jwt_token = request.headers.get('Authorization').split('Bearer ')[1]
        # if not jwt_token:
        #     return jsonify({'error': 'JWT token not found in headers.'}), 400


        profile_data = request.json.get('profile')
        movie_data = request.json.get('movies')

        required_fields = ['profileId', 'profileRatings']
        for field in required_fields:
            if field not in profile_data:
                # print({'error': f'Missing required field: {field}'})

                return jsonify({'error': f'Missing required field: {field}'}), 400

        # movie_data = pd.read_csv("movies.csv", sep=";")
        # movie_data = get_movies(jwt_token)

        X, W, b = train_model(profile_data, movie_data)

        predicted_ratings = tf.matmul(X, tf.transpose(W)) + b
        top_movie_indices = tf.argsort(predicted_ratings, axis=0, direction='DESCENDING')[:25]
        recommended_movies = movie_data.iloc[top_movie_indices.numpy().flatten()]["title"].values

        return jsonify({'recommendations': recommended_movies}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


def find_similar_movies(movie_name, movie_data, num_similar_movies=25):
    # movie_index = movie_data.index[movie_data['title'] == movie_name["title"]].tolist()[0]

    # # movie_features = movie_data['genres'] + ' ' + movie_data['keywords']
    # # tfidf_vectorizer = TfidfVectorizer()
    # # tfidf_matrix = tfidf_vectorizer.fit_transform(movie_features)

    # # cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    # # similar_movie_indices = cosine_similarities.argsort()[-num_similar_movies-1:-1][::-1]

    # # similar_movie_titles = movie_data.iloc[similar_movie_indices]['title'].values

    # movie_features = movie_data['genres'] + ' ' + movie_data['keywords'] + ' ' + movie_data['ratings']

    # rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}

    # def extract_ratings(rating_string):
    #     ratings = rating_string.split('|')
    #     numerical_ratings = [rating_mapping[r.split(',')[1]] for r in ratings]
    #     return sum(numerical_ratings) / len(numerical_ratings) if len(numerical_ratings) > 0 else 0

    # movie_data['ratings'] = movie_data['ratings'].apply(extract_ratings)

    # tfidf_vectorizer = TfidfVectorizer()

    # tfidf_matrix = tfidf_vectorizer.fit_transform(movie_features)

    # cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    # similar_movie_indices = cosine_similarities.argsort()[-num_similar_movies-1:-1][::-1]

    # similar_movie_titles = movie_data.iloc[similar_movie_indices]['title'].values

    # return similar_movie_titles
    movie_index = next((i for i, movie in enumerate(movie_data) if movie['title'] == movie_name['title']), None)
    if movie_index is None:
        return []

    def extract_ratings(ratings_list):
        return ', '.join([rating['rating'] for rating in ratings_list])

    movie_features = [', '.join(movie['genres']) + ', ' + ', '.join(movie['keywords']) + ', ' + extract_ratings(movie['ratings']) for movie in movie_data]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(movie_features)

    cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    similar_movie_indices = cosine_similarities.argsort()[-num_similar_movies-1:-1][::-1]

    similar_movie_titles = [movie_data[i]['title'] for i in similar_movie_indices]

    return similar_movie_titles

@movie_recommender.route('/similar_movies', methods=['POST'])
def get_similar_movies():
    try:
        # Retrieve JWT token from request headers
        # jwt_token = request.headers.get('Authorization').split('Bearer ')[1]

        # if not jwt_token:
        #     return jsonify({'error': 'JWT token not found in headers.'}), 400

        movie_name = request.json.get('movie_name')

        if not movie_name:
            return jsonify({'error': 'Movie name is required.'}), 400

        movie_data = request.json.get('movies')
        # similar_movies = find_similar_movies(movie_name, movie_data)

        similar_movies = find_similar_movies(movie_name, movie_data)

        # similar_movies_list = similar_movies.tolist()

        return jsonify({'similar_movies': similar_movies}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    movie_recommender.run(debug=True)
