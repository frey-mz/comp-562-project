import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from transformation import get_dataset


class LoggyReggy:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Clip z to prevent overflow in exp(-z)
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
    
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        print(f"Starting training for {self.n_iterations} iterations...")
        for i in range(self.n_iterations):
            linear_model = np.dot(x, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
        
            # Compute gradients
            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                # Optional: Calculate and print loss (Cross-Entropy)
                # Add epsilon to prevent log(0)
                epsilon = 1e-9
                cost = (-1 / n_samples) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
                print(f"Iteration {i+1}/{self.n_iterations}, Cost: {cost:.4f}")
        print("Training finished.")

    def predict(self, X): # Use capital X here as it's typically the test set
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        # Return class labels (0 or 1)
        y_predicted_cls = [1 if i >= 0.5 else 0 for i in y_pred]
        return np.array(y_predicted_cls)

print("Loading and preparing dataset...")
# load classifier
df = get_dataset()
print("Dataset loaded. Vectorizing...")
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df['problem']).toarray()
y = df['solvable'].values
print(f"Vectorization complete. Features: {X.shape[1]}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# train
print("Initializing model...")
model = LoggyReggy(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# eval
print("Evaluating model...")
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")