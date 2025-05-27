# my-streamlit-app/streamlit_app.py

import os
import pickle
import time
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.parse import quote
import logging
import random
import streamlit as st # NEW: Import Streamlit

# --- Configuration & Logging ---
# Streamlit handles basic logging, but you can configure more if needed
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings
# tf.get_logger().setLevel('ERROR')

MODEL_PATH = 'problem_recommender.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_SEQUENCE_LEN = 664

# --- Define Rating Range Buffers for OUTPUT filtering ---
RATING_LOWER_BUFFER = 100
RATING_UPPER_BUFFER = 400
MIN_PROBLEM_RATING = 800

# --- Caching resources for Streamlit ---
# Use st.cache_resource to load heavy resources like models only once
@st.cache_resource
def load_resources():
    """Loads the pre-trained model and tokenizer."""
    try:
        # Suppress TensorFlow GPU logging if no GPU found
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            tf.get_logger().setLevel('ERROR')

        model_loaded = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer_loaded = pickle.load(f)
        
        # Determine model input length dynamically if possible, or use MAX_SEQUENCE_LEN - 1
        model_input_length = model_loaded.input_shape[1] if model_loaded.input_shape else MAX_SEQUENCE_LEN - 1
        
        st.success("Model and Tokenizer loaded successfully!")
        return model_loaded, tokenizer_loaded, model_input_length
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop() # Stop the app if resources can't be loaded

model, tokenizer, MODEL_INPUT_LENGTH = load_resources()


# --- Problem Recommender Logic (Adapted for Streamlit) ---

def get_user_data(handle):
    user_info_url = f"https://codeforces.com/api/user.info?handles={quote(handle)}"
    submissions_url = f"https://codeforces.com/api/user.status?handle={quote(handle)}&from=1"

    try:
        user_info_resp = requests.get(user_info_url, timeout=10)
        user_info_resp.raise_for_status()
        user_info = user_info_resp.json()
        if user_info['status'] != 'OK' or not user_info['result']:
            return None, "User not found or API error: " + (user_info.get('comment', ''))

        rating = user_info['result'][0].get('rating')
        if rating is None:
            return None, f"User {handle} does not have a rating."

        submissions_resp = requests.get(submissions_url, timeout=30)
        submissions_resp.raise_for_status()
        submissions = submissions_resp.json()
        if submissions['status'] != 'OK':
            return None, "Error fetching submissions."

        solved_problems = set()
        for s in submissions['result']:
            if s['verdict'] == 'OK' and 'contestId' in s and 'problemsetName' not in s:
                problem_id = f"{s['contestId']}-{s['problem']['index']}"
                solved_problems.add(problem_id)

        return rating, solved_problems
    except requests.exceptions.RequestException as e:
        return None, f"Network or API error: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def get_all_problems():
    problems_url = "https://codeforces.com/api/problemset.problems"
    try:
        resp = requests.get(problems_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data['status'] == 'OK':
            return data['problems'], data['problemStatistics']
        return [], []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching problems from Codeforces API: {e}")
        return [], []

def recommend_problems_streamlit(handle):
    start_time = time.time()
    
    # 1. Fetch user data
    user_current_rating, user_data_error = get_user_data(handle)
    if user_current_rating is None:
        return user_data_error, None, None, None, None

    # 2. Prepare input for model
    all_problems, problem_stats = get_all_problems()
    
    # Filter problems that are rated and within a reasonable range (e.g., 800-3500)
    rated_problems = [p for p in all_problems if 'rating' in p and p['rating'] is not None and 800 <= p['rating'] <= 3500]
    
    # Extract tags from rated problems
    all_tags = set()
    for p in rated_problems:
        all_tags.update(p.get('tags', []))
    
    # Convert tags to sequences for tokenizer
    user_tags_sequence = tokenizer.texts_to_sequences([list(all_tags)])[0]
    padded_sequence = pad_sequences([user_tags_sequence], maxlen=MODEL_INPUT_LENGTH, padding='post')[0]

    # Reshape for model prediction
    model_input = np.array([padded_sequence])

    # 3. Predict tags
    predicted_probs = model.predict(model_input)[0]
    
    # Get top N predicted tags (e.g., top 10 or 20)
    # You might want to adjust this 'top_n' based on model performance
    top_n = 20
    top_tag_indices = np.argsort(predicted_probs)[-top_n:][::-1]
    predicted_tags_features = [tokenizer.index_word[i] for i in top_tag_indices if i in tokenizer.index_word]

    # Filter predicted tags to common Codeforces tags if necessary, or just use as is
    
    # 4. Filter unsolved problems by predicted tags and rating
    
    # Calculate problem rating range dynamically
    rating_floor = max(MIN_PROBLEM_RATING, user_current_rating - RATING_LOWER_BUFFER)
    rating_ceil = user_current_rating + RATING_UPPER_BUFFER

    potential_recommendations = []
    for problem in rated_problems:
        problem_id = f"{problem['contestId']}-{problem['index']}"
        if problem_id in user_data_error: # user_data_error actually contains solved problems set
            continue # Skip already solved problems

        if 'rating' in problem and problem['rating'] is not None:
            problem_rating = problem['rating']
            if rating_floor <= problem_rating <= rating_ceil:
                # Check for overlap with predicted tags
                problem_tags = set(problem.get('tags', []))
                if any(tag in predicted_tags_features for tag in problem_tags):
                    potential_recommendations.append(problem)
    
    # Sort by rating to show easier problems first, then by problem ID
    potential_recommendations.sort(key=lambda p: (p['rating'], p['contestId'], p['index']))

    # Randomly select a few if too many, or just take the first N
    num_recommendations = 10
    if len(potential_recommendations) > num_recommendations:
        # Optional: diversify recommendations by picking from different rating bands
        # or just take the top 'num_recommendations'
        final_recommendations = random.sample(potential_recommendations, num_recommendations)
    else:
        final_recommendations = potential_recommendations

    # 5. Format output
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    response_str = f"## Codeforces Problem Recommendations for {handle}\n"
    response_str += f"**Current Rating:** {user_current_rating}\n"
    response_str += f"**Target Rating Range for Problems:** [{rating_floor}-{rating_ceil}]\n"
    response_str += f"**Processing Time:** {processing_time} seconds\n\n"
    response_str += f"**Predicted Tag Features:** {', '.join(predicted_tags_features)}\n\n"

    if final_recommendations:
        response_str += "**Recommended Problems:**\n"
        for i, rec in enumerate(final_recommendations):
            problem_url = f"https://codeforces.com/problemset/problem/{rec['contestId']}/{rec['index']}"
            response_str += (f"{i+1}. **[{rec['name']} (Rating: {rec['rating']})]({problem_url})**\n"
                             f"   _Tags:_ {', '.join(rec.get('tags', []))}\n")
    else:
        response_str += (f"**No suitable unsolved problems found matching the predicted tags "
                         f"within the rating range [{rating_floor}-{rating_ceil}].**\n"
                         f"This might be due to: \n"
                         f"- Limited unsolved problems in the predicted tags/rating range.\n"
                         f"- User has solved most relevant problems.\n"
                         f"- The model's predictions not aligning with available problems.\n"
                        )
    return response_str, user_current_rating, predicted_tags_features, final_recommendations, processing_time


# --- Streamlit UI ---
st.set_page_config(page_title="CF Problem Recommender", page_icon="💡")

st.title("Codeforces Problem Recommender")
st.markdown("Enter a Codeforces handle to get personalized problem recommendations based on your past performance and predicted tags.")

handle = st.text_input("Codeforces Handle", placeholder="e.g., tourist, feecat, Petr")

if st.button("Get Recommendations"):
    if handle:
        with st.spinner("Fetching data and generating recommendations..."):
            recommendation_markdown, rating, predicted_tags, recommendations_list, time_taken = recommend_problems_streamlit(handle)
            st.markdown(recommendation_markdown)
    else:
        st.error("Please enter a Codeforces handle.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow.")