from flask import Blueprint, request, jsonify
import pickle

recommender = Blueprint('recommender', __name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define an endpoint for movie recommendations
@recommender.route('/recommend', methods=['POST'])
def recommend_movies():
    try:
        # Get user preferences from the request
        user_preferences = request.json

        # Perform any necessary preprocessing on user preferences

        # Make predictions using the model
        recommended_movies = model.predict(user_preferences)

        # Return the recommended movies as a JSON response
        return jsonify({'recommended_movies': recommended_movies.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    recommender.run(debug=True)  # Run the Flask app
