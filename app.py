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
import time

# ✅ Fix: Define NLTK Data Directory
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(NLTK_DIR):
    os.makedirs(NLTK_DIR)
nltk.data.path.append(NLTK_DIR)

# ✅ Ensure NLTK Data is Available
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)

# ✅ Custom Tokenizer to Avoid Errors
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def custom_word_tokenize(text):
    return tokenizer.tokenize(text)

# ✅ Load Stopwords Properly
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
    st.error("❌ Model files not found. Please upload the correct model files.")

# ✅ Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    words = custom_word_tokenize(text)  
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
st.set_page_config(page_title="🚀 Fake Review Detector AI", page_icon="📝", layout="wide")

# ✅ 🎨 3D Glassmorphism UI & Animated Background
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

        body {
            font-family: 'Orbitron', sans-serif;
            background: linear-gradient(135deg, #141E30, #243B55);
            animation: gradientBG 10s infinite alternate;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }

        .title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #00E5FF;
            text-shadow: 0 0 20px #00E5FF, 0 0 40px #00E5FF;
        }

        .stButton button {
            background: linear-gradient(135deg, #FF0080, #6A00F4);
            color: white;
            font-size: 20px;
            padding: 14px;
            border-radius: 12px;
            transition: 0.3s ease-in-out;
            box-shadow: 0 0 20px #FF0080;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #6A00F4, #FF0080);
            box-shadow: 0 0 25px #6A00F4;
            transform: scale(1.05);
        }

        .glass-box {
            background: rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: white;
        }

        .review-box {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 12px;
            font-size: 18px;
            text-align: center;
            color: white;
            box-shadow: 0 0 10px #00E5FF;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ App Title
st.markdown("<h1 class='title'>📝 Fake Review Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🚀 Made by <b>Abdul Rahman Baig</b></h4>", unsafe_allow_html=True)

# ✅ Live Statistics
if "total_reviews" not in st.session_state:
    st.session_state.total_reviews = 0
if "real_reviews" not in st.session_state:
    st.session_state.real_reviews = 0
if "fake_reviews" not in st.session_state:
    st.session_state.fake_reviews = 0

st.markdown(f"<div class='glass-box'>📊 **Total Reviews Analyzed:** {st.session_state.total_reviews}</div>", unsafe_allow_html=True)

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

            st.session_state.total_reviews += 1
            if prediction == 1:
                st.session_state.fake_reviews += 1
                st.markdown(f"<div class='review-box'>❌ **Fake Review Detected!** 😡 (Confidence: {confidence}%)</div>", unsafe_allow_html=True)
            else:
                st.session_state.real_reviews += 1
                st.markdown(f"<div class='review-box'>✅ **Real Review!** 🎉 (Confidence: {confidence}%)</div>", unsafe_allow_html=True)

            # ✅ Animated Pie Chart
            fig, ax = plt.subplots()
            ax.pie([st.session_state.real_reviews, st.session_state.fake_reviews], labels=["Real Reviews", "Fake Reviews"], colors=["green", "red"], autopct='%1.1f%%', startangle=90)
            ax.set_title("Fake vs. Real Reviews")
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
        with open("app_reviews.json", "a") as f:
            json.dump({"name": reviewer_name, "review": app_review}, f)
            f.write("\n")
        st.success("✅ Thank you for your feedback!")

st.markdown("---")
