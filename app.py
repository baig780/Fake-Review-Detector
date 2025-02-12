import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Download necessary NLTK data
import nltk
import os
import ssl

# Fix SSL issue (sometimes needed on cloud servers)
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Ensure NLTK data is downloaded
nltk_data_path = "/home/appuser/nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)
import nltk
import os

# Ensure the correct NLTK data directory exists
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Set NLTK's data path manually
nltk.data.path.append(nltk_data_path)

# Force download of the correct resources
import nltk
import os

# Define the NLTK download directory
NLTK_DIR = "/home/appuser/nltk_data"
if not os.path.exists(NLTK_DIR):
    os.makedirs(NLTK_DIR)

# Set the NLTK path
nltk.data.path.append(NLTK_DIR)

# Download necessary datasets
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)

# Ensure 'punkt' is loaded correctly
from nltk.tokenize import word_tokenize
)


nltk.download("stopwords")
nltk.download("omw-1.4")  # Optional: For wordnet
nltk.download("averaged_perceptron_tagger")  # Optional: If needed for POS tagging

nltk.download("stopwords")

# Load trained models and vectorizer
model_options = {
    "Logistic Regression": "fake_review_detector.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl"
}

vectorizer = joblib.load("tfidf_vectorizer.pkl")
current_model_name = "Logistic Regression"
model = joblib.load(model_options[current_model_name])

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Function to analyze sentiment
def analyze_sentiment(prob):
    if prob > 0.7:
        return "😃 Positive"
    elif 0.4 <= prob <= 0.7:
        return "😐 Neutral"
    else:
        return "😠 Negative"

# Set Streamlit page config
st.set_page_config(page_title="Fake Review Detector", page_icon="📝", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
        }
        .stTextArea textarea {
            font-size: 18px;
            height: 120px !important;
        }
        .stButton button {
            background-color: #FF5733;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 8px;
        }
        .result-box {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("📝 Fake Review Detector AI")
st.write("🔍 Enter a review below to check if it's **Real or Fake** using AI!")

# Dark mode toggle
dark_mode = st.checkbox("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
            body { background-color: #222; color: white; }
            .stTextArea textarea { background-color: #444; color: white; }
            .stButton button { background-color: #00adb5; }
            .result-box { background-color: #333; }
        </style>
        """,
        unsafe_allow_html=True
    )

# Model selection
selected_model = st.selectbox("Select a Model:", list(model_options.keys()))
if selected_model != current_model_name:
    model = joblib.load(model_options[selected_model])
    current_model_name = selected_model

# User input
user_review = st.text_area("✍️ Type your review here:")

if st.button("Check Review 🔍"):
    if user_review.strip():
        try:
            cleaned_review = clean_text(user_review)
            transformed_review = vectorizer.transform([cleaned_review])
            prediction = model.predict(transformed_review)[0]
            prob = model.predict_proba(transformed_review)[0]
            confidence = round(max(prob) * 100, 2)

            st.markdown("---")
            sentiment = analyze_sentiment(prob[1])

            if prediction == 1:
                st.error(f"❌ This is a **Fake Review!** 😡 (Confidence: {confidence}%)\n\nSentiment: {sentiment}")
            else:
                st.success(f"✅ This is a **Real Review!** 🎉 (Confidence: {confidence}%)\n\nSentiment: {sentiment}")

            # Confidence Score Visualization
            fig, ax = plt.subplots()
            ax.bar(["Real Review", "Fake Review"], prob * 100, color=["green", "red"])
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Confidence Levels")
            st.pyplot(fig)

            # Option to download result
            result_text = f"Review: {user_review}\nPrediction: {'Fake Review' if prediction == 1 else 'Real Review'}\nConfidence: {confidence}%\nSentiment: {sentiment}"
            st.download_button(label="📥 Download Result", data=result_text, file_name="review_result.txt", mime="text/plain")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

st.markdown("---")
st.markdown("<h5 style='text-align: center;'>🔥 Built with ❤️ using Streamlit & AI 🔥</h5>", unsafe_allow_html=True)
