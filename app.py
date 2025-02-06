from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import nltk
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model, tokenizer, and intents
model = tf.keras.models.load_model("gru_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("texts.pkl", "rb"))
classes = pickle.load(open("labels.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

# Function to preprocess input text
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert input sentence into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array([bag])

# Predict intent
def predict_intent(text):
    bow = bag_of_words(text)
    bow = bow.reshape((1, 1, bow.shape[1]))  # Reshape for GRU input
    prediction = model.predict(bow)
    intent_index = np.argmax(prediction)
    return classes[intent_index]

# Get chatbot response
def get_response(intent):
    for intent_data in intents["intents"]:
        if intent_data["tag"] == intent:
            return random.choice(intent_data["responses"])
    return "I'm sorry, I don't understand."

# API route for chatbot
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    intent = predict_intent(user_message)
    response = get_response(intent)
    return jsonify({"response": response})

# Web interface
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
