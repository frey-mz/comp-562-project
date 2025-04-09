import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np


class LoggyReggy:
    def __init__(self, LinnyReggy=0.1, LIFETIMES=1000):
        self.LinnyReggy = LinnyReggy
        self.LIFETIMES = LIFETIMES

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, x, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.LIFETIMES):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
        
            dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(X)) * np.sum(y_pred - y)

            self.weights -= self.LinnyReggy * dw
            self.bias -= self.LinnyReggy * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i>= 0.5 else 0 for i in y_pred]

# load classifier
df = ...
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df['prob']).toarray()
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
model = LoggyReggy(LinnyReggy=0.1, LIFETIMES=1000)
model.fit(X_train, y_train)

# eval
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"acc: {accuracy * 100:.2f}%")