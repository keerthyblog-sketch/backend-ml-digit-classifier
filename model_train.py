import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

# 1. Load the MNIST dataset
print("Loading MNIST dataset... this might take a minute.")
# fetch_openml returns data in a bunch-like object
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist["data"], mnist["target"]

# 2. Preprocess the data
# Normalize pixel values to [0, 1]
X = X / 255.0

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. Initialize and train the model
# Using a simple Multi-layer Perceptron (MLP) Classifier
print("Training MLP Classifier...")
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, alpha=1e-4,
                      solver='sgd', verbose=10, random_state=1,
                      learning_rate_init=.1)

model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# 5. Save the trained model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
