# -*- coding: utf-8 -*-
"""training mental health bot .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oeGtp-3ue3I-Nt3Wl7Wtp_UqqthjpkZ8
"""

from google.colab import drive
drive.mount('/content/drive')

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import json
import numpy as np
import random
import pickle

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents file
data_file = open('/content/drive/MyDrive/bot/intents.json').read()
intents = json.loads(data_file)

nltk.download('punkt_tab')

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Tokenizing and lemmatizing
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# ... (Your existing code)

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

# Save words and classes for later use
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Find the maximum length of bag
max_len = max(len(item[0]) for item in training)

# Pad shorter bags with zeros
for item in training:
    if len(item[0]) < max_len:
        item[0] += [0] * (max_len - len(item[0]))

training = np.array(training, dtype=object)

# Shuffle and split into training and testing
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Create the ANN model
model_ann = Sequential()

# Input layer with 128 neurons and dropout
model_ann.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.01)))
model_ann.add(Dropout(0.5))  # Add Dropout to prevent overfitting

# Hidden layer with 64 neurons and dropout
model_ann.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model_ann.add(Dropout(0.5))  # Add Dropout to prevent overfitting

# Output layer with softmax activation (for multi-class classification)
model_ann.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model with Adam optimizer
optimizer = Adam(learning_rate=0.001)
model_ann.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early stopping to prevent overfitting (monitor validation loss)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
hist_ann = model_ann.fit(np.array(train_x), np.array(train_y),
                         epochs=200, batch_size=8,
                         validation_data=(np.array(test_x), np.array(test_y)),
                         callbacks=[early_stopping])

# Evaluate the ANN model on the test data
test_loss, test_accuracy = model_ann.evaluate(np.array(test_x), np.array(test_y), verbose=1)

# Print the accuracy and loss
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np

# Reshape data for LSTM
train_x_lstm = np.array(train_x).reshape((len(train_x), 1, len(train_x[0])))
test_x_lstm = np.array(test_x).reshape((len(test_x), 1, len(test_x[0])))

# Define LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(128, input_shape=(1, len(train_x[0])), dropout=0.3))  # Reduced dropout
model_lstm.add(Dense(64, activation='relu'))  # Additional dense layer
model_lstm.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model (Use Adam instead of SGD)
optimizer = Adam(learning_rate=0.001)
model_lstm.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train LSTM Model
hist_lstm = model_lstm.fit(train_x_lstm, np.array(train_y),
                           epochs=100, batch_size=32,  # Increased batch size
                           validation_data=(test_x_lstm, np.array(test_y)),
                           callbacks=[early_stopping])

# Evaluate Model
test_loss, test_accuracy = model_lstm.evaluate(test_x_lstm, np.array(test_y), verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
import numpy as np

# Reshaping data for GRU: (samples, timesteps, features)
train_x_gru = np.array(train_x)
train_x_gru = train_x_gru.reshape((train_x_gru.shape[0], 1, train_x_gru.shape[1]))  # Reshaping for GRU
test_x_gru = np.array(test_x)
test_x_gru = test_x_gru.reshape((test_x_gru.shape[0], 1, test_x_gru.shape[1]))  # Reshaping for GRU

# GRU Model Architecture
model_gru = Sequential()

# First GRU layer with 128 units, return sequences to pass to the next GRU layer
model_gru.add(GRU(128, input_shape=(train_x_gru.shape[1], train_x_gru.shape[2]), return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

# Second GRU layer with 64 units
model_gru.add(GRU(64, dropout=0.5, recurrent_dropout=0.5))

# Output layer with softmax activation
model_gru.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)  # You can switch this to SGD if preferred
model_gru.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the GRU model
hist_gru = model_gru.fit(np.array(train_x_gru), np.array(train_y),
                         epochs=200, batch_size=8,
                         validation_data=(np.array(test_x_gru), np.array(test_y)),
                         callbacks=[early_stopping])

# Evaluate the model on the test data
test_loss_gru, test_accuracy_gru = model_gru.evaluate(np.array(test_x_gru), np.array(test_y), verbose=1)

# Print the accuracy and loss
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

import matplotlib.pyplot as plt

# Model names and their respective accuracies
models = ["ANN", "LSTM", "GRU"]
accuracies = [52, 63, 78]  # Accuracy values

# Set the figure size
plt.figure(figsize=(8, 5))

# Create a bar chart
plt.bar(models, accuracies, color=['blue', 'green', 'red'])

# Add labels and title
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

# Display accuracy values on bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc}%", ha='center', fontsize=12, fontweight='bold')

# Show the plot
plt.ylim(0, 100)  # Set y-axis limit
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Model names and their respective accuracies
models = ["ANN", "LSTM", "GRU"]
accuracies = [52, 63, 78]  # Accuracy values

# Create a DataFrame for the table
df = pd.DataFrame({"Model": models, "Accuracy (%)": accuracies})

# Display the table
print(df)

# Plot Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])

# Add labels and title
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

# Display accuracy values on bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc}%", ha='center', fontsize=12, fontweight='bold')

# Show the plot
plt.ylim(0, 100)  # Set y-axis limit
plt.show()

model_gru.save("gru_model.h5")
