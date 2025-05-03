# Data format
# {"problem": "", "correct": 0|1|2|3, "answer": "", "solution": "", "id": "", "cost": [0.0, 0.0, 0.0, 0.0]}
# correct denotes tier of model that was able to solve the problem
# need a multiclass classifier to predict kind of model that will solve correctly.

# imagine there is a json file called "problems.jsonl"

# load data
import keras
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
load_dotenv()
import voyageai
from sklearn.model_selection import train_test_split

vo = voyageai.Client()

df = pd.read_json("problems.jsonl", lines=True)

# Save embeddings to a file if the file doesn't exist/is empty, otherwise embed.

if os.path.exists("embeddings.csv"):
    X = np.loadtxt("embeddings.csv", delimiter=',')
else:
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

y = df['correct'].values

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
print("Initializing model...")
model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)

# eval
print("Evaluating model...")
predictions = model.predict(X_test)
accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# save predictions and accuracy in text
with open("results.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Predictions: {predictions}\n")

print("Model training complete.")

if __name__ == "__main__":
    main()
