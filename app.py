import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Load trained model and vectorizer
try:
    model = joblib.load("fake_review_detector.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

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
            if prediction == 1:
                st.error(f"❌ This is a **Fake Review!** 😡 (Confidence: {confidence}%)")
            else:
                st.success(f"✅ This is a **Real Review!** 🎉 (Confidence: {confidence}%)")

            # Option to download result
            result_text = f"Review: {user_review}\nPrediction: {'Fake Review' if prediction == 1 else 'Real Review'}\nConfidence: {confidence}%"
            st.download_button(label="📥 Download Result", data=result_text, file_name="review_result.txt", mime="text/plain")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

st.markdown("---")
st.markdown("<h5 style='text-align: center;'>🔥 Built with ❤️ using Streamlit & AI 🔥</h5>", unsafe_allow_html=True)
