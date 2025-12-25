import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
model = pickle.load(open("model/svm_genre_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

# -------------------------------
# Text Preprocessing
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Movie Genre Classifier", layout="centered")

st.title("🎬 Movie Genre Classification App")
st.write("Predict the **genre of a movie** using its plot description.")

plot_input = st.text_area(
    "Enter Movie Plot Description:",
    height=200,
    placeholder="A fearless police officer fights crime and corruption in the city..."
)

if st.button("Predict Genre"):
    if plot_input.strip() == "":
        st.warning("Please enter a movie plot description.")
    else:
        clean_plot = preprocess_text(plot_input)
        plot_vector = vectorizer.transform([clean_plot])
        prediction = model.predict(plot_vector)

        st.success(f"🎯 Predicted Genre: **{prediction[0]}**")
