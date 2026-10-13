import streamlit as st
import joblib
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

st.title("News Text Classifier")
st.write("Classify a news snippet into World, Sports, Business, or Sci/Tech.")

labels = ['World', 'Sports', 'Business', 'Sci/Tech']

model_choice = st.selectbox("Choose a model", ["Logistic Regression", "CNN", "DistilBERT"])

text_input = st.text_area("Enter news text here")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        if model_choice == "Logistic Regression":
            lr_model = joblib.load("lr_model.pkl")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            X = vectorizer.transform([text_input])
            pred = lr_model.predict(X)[0]
            st.success(f"Prediction: {labels[pred]}")

        elif model_choice == "CNN":
            cnn_model = load_model("cnn_model.keras")
            with open("tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
            seq = tokenizer.texts_to_sequences([text_input])
            padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
            pred = cnn_model.predict(padded).argmax(axis=1)[0]
            st.success(f"Prediction: {labels[pred]}")

        elif model_choice == "DistilBERT":
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert_model")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert_model")
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=100)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = outputs.logits.argmax(dim=1).item()
            st.success(f"Prediction: {labels[pred]}")
