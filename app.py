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

# ✅ Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

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
    words = [word for word in words if word not in stopwords.words("english")]
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

# ✅ Custom CSS for styling
st.markdown("""
    <style>
        body { font-family: 'Arial', sans-serif; background-color: #f5f5f5; }
        .stTextArea textarea { font-size: 18px; height: 120px !important; }
        .stButton button { background-color: #FF5733; color: white; font-size: 18px; padding: 10px; border-radius: 8px; }
        .result-box { background-color: #ffffff; padding: 15px; border-radius: 10px; font-size: 20px; text-align: center; font-weight: bold; }
        .review-box { background: linear-gradient(135deg, #FF416C, #FF4B2B); padding: 15px; border-radius: 12px; font-size: 18px; text-align: center; color: white; }
        .stSelectbox select { font-size: 16px; }
        .stTextInput input { font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

# ✅ App Title
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>📝 Fake Review Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🚀 Made by <b>Abdul Rahman Baig</b></h4>", unsafe_allow_html=True)

# ✅ Dark mode toggle
dark_mode = st.checkbox("🌙 Enable Dark Mode")
if dark_mode:
    st.markdown("<style>body { background-color: #222; color: white; }</style>", unsafe_allow_html=True)

# ✅ Model selection
selected_model = st.selectbox("Select a Model:", list(model_options.keys()))
if selected_model != current_model_name:
    model = joblib.load(model_options[selected_model])
    current_model_name = selected_model

# ✅ User input
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
