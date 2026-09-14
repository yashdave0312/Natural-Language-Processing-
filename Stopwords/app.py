import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK dependencies are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model, vectorizer, and label encoder
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    text = ''.join([i for i in text if i.isascii()])
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

st.title("🎭 Emotion Detection App")
st.write("Type a sentence or thought below to analyze its underlying emotion.")

user_input = st.text_area("Input Text:")

if st.button("Predict Emotion"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        predicted_emotion = le.inverse_transform(prediction)[0]
        
        st.success(f"Predicted Emotion: **{predicted_emotion.capitalize()}**")
    else:
        st.warning("Please enter some text to analyze.")