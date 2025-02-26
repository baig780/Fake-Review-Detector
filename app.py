import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

import nltk
import os

import nltk
import os

# ✅ Define a permanent directory for NLTK data
NLTK_DIR = "/home/appuser/nltk_data"

# ✅ Ensure the directory exists
if not os.path.exists(NLTK_DIR):
    os.makedirs(NLTK_DIR)

# ✅ Set the NLTK data path manually
nltk.data.path.append(NLTK_DIR)

# ✅ Force download necessary resources and save them in the correct directory
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)

# ✅ Manually ensure 'punkt' is loaded correctly
try:
    from nltk.tokenize import word_tokenize
    word_tokenize("Test tokenization")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DIR)


# ✅ Load trained models and vectorizer
model_options = {
    "Logistic Regression": "fake_review_detector.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl"
}

vectorizer = joblib.load("tfidf_vectorizer.pkl")
current_model_name = "Logistic Regression"
model = joblib.load(model_options[current_model_name])

# ✅ Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ✅ Function to analyze sentiment
def analyze_sentiment(prob):
    if prob > 0.7:
        return "😃 Positive"
    elif 0.4 <= prob <= 0.7:
        return "😐 Neutral"
    else:
        return "😠 Negative"

# ✅ Set Streamlit page config
st.set_page_config(page_title="Fake Review Detector", page_icon="📝", layout="centered")

# ✅ Custom CSS for a Futuristic UI
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #1E1E1E, #0F0F0F);
            color: white;
        }
        .stTextArea textarea {
            font-size: 18px; height: 120px !important;
            background: #1E1E1E; color: white; border-radius: 8px;
        }
        .stButton button {
            background: linear-gradient(135deg, #FF416C, #FF4B2B);
            color: white; font-size: 18px; padding: 10px;
            border-radius: 8px; box-shadow: 0px 0px 15px #FF4B2B;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #FF4B2B, #FF416C);
            box-shadow: 0px 0px 20px #FF416C;
        }
        .result-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px; border-radius: 15px;
            font-size: 22px; text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

# ✅ App Title
st.markdown("<h1 style='text-align: center; color: #00E5FF;'>📝 Fake Review Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🚀 Made by <b>Abdul Rahman Baig</b></h4>", unsafe_allow_html=True)

# ✅ Dark Mode Toggle
dark_mode = st.checkbox("🌙 Enable Dark Mode")
if dark_mode:
    st.markdown("<style>body { background-color: #222; color: white; }</style>", unsafe_allow_html=True)

# ✅ Model Selection
selected_model = st.selectbox("Select a Model:", list(model_options.keys()))
if selected_model != current_model_name:
    model = joblib.load(model_options[selected_model])
    current_model_name = selected_model

# ✅ User Input Section
st.markdown("### 🔍 Enter a Review to Analyze")
user_review = st.text_area("✍️ Type your review here:")

if st.button("🚀 Analyze Review Now"):
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
                st.markdown(f"<div class='result-box'>❌ **Fake Review Detected!** 😡 (Confidence: {confidence}%)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-box'>✅ **Real Review!** 🎉 (Confidence: {confidence}%)</div>", unsafe_allow_html=True)

            # ✅ Confidence Score Visualization
            fig, ax = plt.subplots()
            ax.bar(["Real Review", "Fake Review"], prob * 100, color=["green", "red"])
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Confidence Levels")
            st.pyplot(fig)

            # ✅ Option to download result
            result_text = f"Review: {user_review}\nPrediction: {'Fake Review' if prediction == 1 else 'Real Review'}\nConfidence: {confidence}%\nSentiment: {sentiment}"
            st.download_button(label="📥 Download Result", data=result_text, file_name="review_result.txt", mime="text/plain")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# ✅ Review Submission Section
st.markdown("---")  
st.subheader("📝 Give Your Honest Review About This App")  

reviewer_name = st.text_input("Your Name", "")
app_review = st.text_area("Your Review", "")

if st.button("Submit Review"):
    if reviewer_name.strip() and app_review.strip():
        review_entry = {"name": reviewer_name, "review": app_review}

        # ✅ Load existing reviews
        if os.path.exists("app_reviews.json"):
            with open("app_reviews.json", "r") as f:
                try:
                    review_data = json.load(f)
                except json.JSONDecodeError:
                    review_data = []
        else:
            review_data = []

        review_data.append(review_entry)

        # ✅ Save updated reviews
        with open("app_reviews.json", "w") as f:
            json.dump(review_data, f, indent=4)

        st.success("✅ Thank you for your feedback!")
    else:
        st.warning("⚠️ Please enter your name and review before submitting.")

# ✅ Display All User Reviews
st.markdown("---")  
st.subheader("📢 User Reviews About This App")

try:
    with open("app_reviews.json", "r") as f:
        review_data = json.load(f)

    if review_data:
        for review in review_data[-10:]:  # Show the last 10 reviews
            st.write(f"📝 **{review['name']}**: {review['review']}")
    else:
        st.info("No reviews yet. Be the first to leave feedback! 😊")
except FileNotFoundError:
    st.info("No reviews yet. Be the first to leave feedback! 😊")

st.markdown("---")
st.markdown("<h4 style='text-align: center;'>🔥 Built with ❤️ using Streamlit & AI 🔥</h4>", unsafe_allow_html=True)
