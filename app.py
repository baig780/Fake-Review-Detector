import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# ✅ Fix: Define NLTK Data Directory
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(NLTK_DIR):
    os.makedirs(NLTK_DIR)
nltk.data.path.append(NLTK_DIR)

# ✅ Fix: Ensure NLTK Data is Available Before Running
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)

# ✅ Fix: Custom Tokenizer to Avoid `punkt_tab` Error
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def custom_word_tokenize(text):
    return tokenizer.tokenize(text)

# ✅ Fix: Ensure Stopwords Are Loaded Properly
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DIR)
    stop_words = set(stopwords.words("english"))

# ✅ Load trained models and vectorizer safely
model_options = {
    "Logistic Regression": "fake_review_detector.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl"
}

try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    current_model_name = "Logistic Regression"
    model = joblib.load(model_options[current_model_name])
except FileNotFoundError:
    st.error("❌ Model files not found. Please upload the correct model files to your project directory.")

# ✅ Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = custom_word_tokenize(text)  # ✅ Use Custom Tokenizer
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

# ✅ 🔥 Stunning CSS for Modern UI
st.markdown("""
    <style>
        /* 🌟 Custom Font & Background */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #141E30, #243B55);
            color: white;
            animation: gradientAnimation 10s infinite alternate;
        }

        /* 🔥 Neon Glow for Title */
        .title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #00E5FF;
            text-shadow: 0 0 20px #00E5FF, 0 0 40px #00E5FF;
        }

        /* 🎨 Fancy Buttons */
        .stButton button {
            background: linear-gradient(135deg, #00E5FF, #0096FF);
            color: white;
            font-size: 18px;
            padding: 12px;
            border-radius: 10px;
            transition: 0.3s ease-in-out;
            box-shadow: 0 0 20px #00E5FF;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #0096FF, #00E5FF);
            box-shadow: 0 0 25px #0096FF;
            transform: scale(1.05);
        }

        /* 🟢 Text Input Styling */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }

        /* 🌟 Results Box */
        .result-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 15px;
            font-size: 22px;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
        }

        /* 🔥 Reviews Section */
        .review-box {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 12px;
            font-size: 18px;
            text-align: center;
            color: white;
            box-shadow: 0 0 10px #00E5FF;
        }

        /* 🔥 Gradient Animation */
        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
    </style>
""", unsafe_allow_html=True)

# ✅ App Title
st.markdown("<h1 class='title'>📝 Fake Review Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🚀 Made by <b>Abdul Rahman Baig</b></h4>", unsafe_allow_html=True)

# ✅ Review Checker Section
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
                st.markdown(f"<div class='review-box'>❌ **Fake Review Detected!** 😡 (Confidence: {confidence}%)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='review-box'>✅ **Real Review!** 🎉 (Confidence: {confidence}%)</div>", unsafe_allow_html=True)

            # ✅ Confidence Score Visualization
            fig, ax = plt.subplots()
            ax.bar(["Real Review", "Fake Review"], prob * 100, color=["green", "red"])
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Confidence Levels")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# ✅ User Reviews Section
st.markdown("---")  
st.subheader("📝 Give Your Honest Review About This App")  

reviewer_name = st.text_input("Your Name", "")
app_review = st.text_area("Your Review", "")

if st.button("Submit Review"):
    if reviewer_name.strip() and app_review.strip():
        review_entry = {"name": reviewer_name, "review": app_review}

        with open("app_reviews.json", "a") as f:
            json.dump(review_entry, f)
            f.write("\n")

        st.success("✅ Thank you for your feedback!")

# ✅ Display User Reviews
st.markdown("---")  
st.subheader("📢 User Reviews About This App")

try:
    with open("app_reviews.json", "r") as f:
        for line in f:
            review = json.loads(line)
            st.markdown(f"<div class='review-box'>📝 **{review['name']}**: {review['review']}</div>", unsafe_allow_html=True)
except FileNotFoundError:
    st.info("No reviews yet. Be the first to leave feedback! 😊")
