from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import mysql.connector
import json
import os

movie_recommender = Blueprint('recommend', __name__)

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

db_connection = mysql.connector.connect(
    host=db_host,
    port=db_port,
    user=db_user,
    password=db_password,
    database=db_name
)

def fetch_movie_data():
    cursor = db_connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT m.*,
        JSON_ARRAYAGG(
           JSON_OBJECT(
               'ratingId', r.movie_rating_Id,
               'rating', r.rating,
               'date', r.date,
               'profileId', r.profile_Id
           )
        ) as ratings,
        (SELECT JSON_ARRAYAGG(mg.genres) FROM movie_genres mg WHERE mg.movie_movieid = m.movieid) as genres,
        (SELECT JSON_ARRAYAGG(mk.keywords) FROM movie_keywords mk WHERE mk.movie_movieid = m.movieid) as keywords,
        (SELECT JSON_ARRAYAGG(mc.casts) FROM movie_casts mc WHERE mc.movie_movieid = m.movieid) as casts,
        (SELECT JSON_ARRAYAGG(mco.companies) FROM movie_companies mco WHERE mco.movie_movieid = m.movieid) as companies,
        (SELECT JSON_ARRAYAGG(mw.writers) FROM movie_writers mw WHERE mw.movie_movieid = m.movieid) as writers
        FROM movies m
        LEFT JOIN movie_ratings_join mj ON m.movieid = mj.movieid
        LEFT JOIN movie_ratings r ON mj.movie_rating_id = r.movie_rating_id
        GROUP BY m.movieid
    """)
    movies = cursor.fetchall()
    cursor.close()
    return movies

def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R)
    return(Ynorm, Ymean)

def train_model(user_profile, movie_data, num_features=8, lambda_=0.05, epochs=300):
    rating_mapping = {"dislike": 1, "like": 3, "superlike": 5}

    # Extract user preferences from the user profile
    user_preferences = []
    for rating in user_profile["profileRatings"]:
        rating_numeric = rating_mapping.get(rating["rating"].lower())
        if rating_numeric is not None:
            user_preferences.append({"movieid": rating["movie"]["movieid"], "rating": rating_numeric})

    num_users_r = 1
    num_movies_r = len(movie_data)

    # Initialize matrices Y and R for ratings and indicator variables
    Y_r = np.zeros((num_movies_r, num_users_r))
    for rating in user_preferences:
        movie_index = movie_data.index[movie_data['movieid'] == rating['movieid']][0]
        Y_r[movie_index, 0] = rating['rating']

    R_r = np.ones((num_movies_r, num_users_r))

    # Normalize ratings
    Y_r_norm, Y_r_mean = normalizeRatings(Y_r, R_r)

    # Set Initial Parameters (W, X), use tf.Variable to track these variables initialized using random values.
    tf.random.set_seed(1234) # for consistent results
    W_r = tf.Variable(tf.random.normal((num_users_r,  num_features), dtype=tf.float64), name='W_r')
    X_r = tf.Variable(tf.random.normal((num_movies_r, num_features), dtype=tf.float64), name='X_r')
    b_r = tf.Variable(tf.random.normal((1,          num_users_r), dtype=tf.float64), name='b_r')

    # Instantiate an optimizer. The Adam optimizer is used to minimize the cost function by adjusting the model parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    # Define the cost function for collaborative filtering
    def cofi_cost_func_v(X, W, b, Y, R, lambda_):
        """
        Returns the cost for the collaborative filtering
        Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
        Args:
        X (ndarray (num_movies,num_features)): matrix of item features
        W (ndarray (num_users,num_features)) : matrix of user parameters
        b (ndarray (1, num_users)            : vector of user parameters
        Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
        R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
        lambda_ (float): regularization parameter
        Returns:
        J (float) : Cost
        """
        j = tf.linalg.matmul(X, tf.transpose(W)) + b - Y
        J = 0.5 * tf.reduce_sum((j * R) ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
        return J

    for epoch in range(epochs):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost
        with tf.GradientTape() as tape:
            # Compute the cost (forward pass included in cost)
            cost = cofi_cost_func_v(X_r, W_r, b_r, Y_r_norm, R_r, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        gradients = tape.gradient(cost, [X_r, W_r, b_r])

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(gradients, [X_r,W_r,b_r]))

        # Log periodically.
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Cost: {cost.numpy():.4f}")

    return X_r, W_r, b_r, Y_r_mean

# Variables for caching the trained model, tracking its training status, and the last profile ID used
cached_model = None
is_model_trained = False
last_profile_id = None

@movie_recommender.route('/user_recommendations', methods=['POST'])
def get_recommendations():
    try:
        global cached_model, is_model_trained, last_profile_id

        # Extract profile data from the request
        profile_data = request.json.get('profile')

        # Fetch movie data from the database
        movie_data = fetch_movie_data()
        movie_df = pd.DataFrame(movie_data)

        # Check if required fields are present in the profile data
        required_fields = ['profileId', 'profileRatings']
        for field in required_fields:
            if field not in profile_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # If the profile ID has changed, reset the model cache
        if profile_data['profileId'] != last_profile_id:
            # Reset model cache if the profile has changed
            is_model_trained = False
            last_profile_id = profile_data['profileId']
        # If the model has not been trained for the current profile, train it
        if not is_model_trained:
            X, W, b, Y_r_mean = train_model(profile_data, movie_df)
            cached_model = (X, W, b, Y_r_mean)
            is_model_trained = True
        # Otherwise, use the cached model
        else:
            X, W, b, Y_r_mean = cached_model

        # Compute predicted ratings for all movies
        predicted_ratings = tf.matmul(X, tf.transpose(W)) + b
        # Restore the mean to get normalized predicted ratings
        pm = predicted_ratings + Y_r_mean
        # Get indices of top-rated movies
        top_movie_indices = tf.argsort(pm, axis=0, direction='DESCENDING')[:20]
        # Retrieve details of recommended movies
        recommended_movies = movie_df.iloc[top_movie_indices.numpy().flatten()].to_dict(orient='records')
        # Convert certain attributes from JSON strings to Python lists for better readability
        for movie in recommended_movies:
            movie['casts'] = json.loads(movie['casts'])
            movie['companies'] = json.loads(movie['companies'])
            movie['genres'] = json.loads(movie['genres'])
            movie['keywords'] = json.loads(movie['keywords'])
            movie['ratings'] = json.loads(movie['ratings'])
            movie['writers'] = json.loads(movie['writers'])

        return jsonify({'recommendations': recommended_movies}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


def find_similar_movies(movie_name, movie_data, num_similar_movies=20):
    # Find the index of the specified movie in the movie dataset
    movie_index = movie_data.index[movie_data['title'] == movie_name["title"]].tolist()[0]
    # Combine movie features (genres, keywords, ratings) into a single string representation
    movie_features = [', '.join(movie_data.iloc[i]['genres']) + ', ' + ', '.join(movie_data.iloc[i]['keywords']) + ', ' + str(movie_data.iloc[i]['ratings']) for i in range(len(movie_data))]
    # Apply TF-IDF vectorization to movie features
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(movie_features)
    # Compute cosine similarities between the specified movie and all other movies
    cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    # Get indices of top similar movies
    similar_movie_indices = cosine_similarities.argsort()[-num_similar_movies-1:-1][::-1]
    # Retrieve details of similar movies
    similar_movies = movie_data.iloc[similar_movie_indices].to_dict(orient='records')
    # Convert certain attributes from JSON strings to Python lists for better readability
    for movie in similar_movies:
        movie['casts'] = json.loads(movie['casts'])
        movie['companies'] = json.loads(movie['companies'])
        movie['genres'] = json.loads(movie['genres'])
        movie['keywords'] = json.loads(movie['keywords'])
        movie['ratings'] = json.loads(movie['ratings'])
        movie['writers'] = json.loads(movie['writers'])

    return similar_movies

@movie_recommender.route('/similar_movies', methods=['POST'])
def get_similar_movies():
    try:
        movie_name = request.json.get('movie_name')

        if not movie_name:
            return jsonify({'error': 'Movie name is required.'}), 400

        movie_data = fetch_movie_data()
        movie_df = pd.DataFrame(movie_data)

        similar_movies = find_similar_movies(movie_name, movie_df)

        return jsonify({'similar_movies': similar_movies}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    movie_recommender.run(debug=True)
