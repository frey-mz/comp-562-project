import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
import voyageai

vo = voyageai.Client()

# Load environment variables from .env file
load_dotenv()

# train_test_split
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, learning_rate=0.1, n_iterations=200):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.model = keras.models.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def fit(self, x, y):
        self.model.fit(x, y, epochs=self.n_iterations, verbose=1)
        
    def predict(self, x):
        return self.model.predict(x)
        
    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=1)
    
def main():
    # load data
    df = pd.read_json("validated.jsonl", lines=True)
    if not os.path.exists("embeddings.csv"):
        problems = df['problem'].tolist()
        X = np.empty((0, 1024), dtype=np.float32)
        count = 0
        for i in range(0, len(problems), 128):
            count += 1
            if count >= 1999:
                time.sleep(60)
                count = 0
            X = np.append(X, vo.embed(
                problems[i:i+128], model="voyage-3"
            ).embeddings, axis=0)
            print(f"Processed {i+128} problems...")
        np.savetxt("embeddings.csv", X, delimiter=',')
    else:
        X = np.loadtxt("embeddings.csv", delimiter=',')
    y = df['correct'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # train
    print("Initializing model...")
    model = NeuralNet(learning_rate=0.1, n_iterations=200)
    model.fit(X_train, y_train)
    
    # eval
    print("Evaluating model...")
    predictions = model.predict(X_test)
    predictions = np.round(predictions).astype(int)
    accuracy = np.mean(predictions == y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # save predictions and accuracy in text
    with open("results.txt", "w") as f:
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Predictions: {predictions}\n")
    
    print("Model training complete.")

if __name__ == "__main__":
    main()
