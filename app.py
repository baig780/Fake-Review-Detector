import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import json
import os
import matplotlib.pyplot as plt  # ✅ For Chart Visualization

# ✅ Fix: Define NLTK Data Directory
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(NLTK_DIR):
    os.makedirs(NLTK_DIR)
nltk.data.path.append(NLTK_DIR)

# ✅ Ensure NLTK Data is Available Before Running
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)

# ✅ Custom Tokenizer to Avoid `punkt_tab` Error
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def custom_word_tokenize(text):
    return tokenizer.tokenize(text)

# ✅ Ensure Stopwords Are Loaded Properly
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DIR)
    stop_words = set(stopwords.words("english"))

# ✅ Load trained models and vectorizer
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
st.set_page_config(page_title="Fake Review Detector", page_icon="📝", layout="wide")

# ✅ 🚀 **Super-Futuristic 3D UI with Holographic Effects**
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');
        
        /* 🌟 Animated Neon Background */
        body {
            font-family: 'Orbitron', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            background-size: 300% 300%;
            animation: moveBackground 10s infinite alternate;
            color: white;
        }

        @keyframes moveBackground {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }

        /* 🚀 3D Floating Title */
        .title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #0ff;
            text-shadow: 0px 0px 15px #0ff, 0px 0px 30px #0ff;
        }

        /* 🔥 Holographic Buttons */
        .stButton button {
            background: linear-gradient(90deg, #00E5FF, #00FF87);
            color: white;
            font-size: 20px;
            border-radius: 12px;
            padding: 15px;
            transition: 0.3s;
            box-shadow: 0px 0px 30px #00E5FF;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #00FF87, #00E5FF);
            box-shadow: 0px 0px 40px #00FF87;
            transform: scale(1.1);
        }

        /* 🟢 Glowing Input Fields */
        .stTextArea textarea, .stTextInput input {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 20px;
            border: 2px solid #00E5FF;
            box-shadow: 0px 0px 20px #00E5FF;
        }

        /* 🚀 Holographic Result Box */
        .result-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 15px;
            font-size: 22px;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0px 0px 30px rgba(0, 229, 255, 0.8);
        }
    </style>
""", unsafe_allow_html=True)

# ✅ **Holographic App Title**
st.markdown("<h1 class='title'>📝 Fake Review Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🚀 Made by <b>Abdul Rahman Baig</b></h4>", unsafe_allow_html=True)

# ✅ **Review Checker Section**
st.markdown("### 🔍 Enter a Review to Analyze")
user_review = st.text_area("✍️ Type your review here:")

if st.button("🚀 Analyze Review Now"):
    if user_review.strip():
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

        # ✅ **Chart Visualization for Real vs Fake**
        labels = ["Real Review", "Fake Review"]
        sizes = [prob[0] * 100, prob[1] * 100]
        colors = ["#00E5FF", "#FF5733"]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, shadow=True)
        ax.set_title("Prediction Confidence Levels")
        st.pyplot(fig)

# ✅ **User Reviews Section**
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

# ✅ **Display User Reviews**
st.markdown("---")
st.subheader("📢 User Reviews About This App")

try:
    with open("app_reviews.json", "r") as f:
        for line in f:
            review = json.loads(line)
            st.markdown(f"<div class='result-box'>📝 **{review['name']}**: {review['review']}</div>", unsafe_allow_html=True)
except FileNotFoundError:
    st.info("No reviews yet. Be the first to leave feedback! 😊")
