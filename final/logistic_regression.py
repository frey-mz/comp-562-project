import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
load_dotenv()
import voyageai
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

vo = voyageai.Client()

# Load upsampled training and validation sets (same as neural_net.py)
df = pd.read_json("./HARDMATH/hardmath_output_upsampled_train.jsonl", lines=True).sample(frac=1, random_state=42)
df2 = pd.read_json("./NEURIPS/neurips_output_upsampled_train.jsonl", lines=True).sample(frac=1, random_state=42)
df3 = pd.read_json("./DEEPMIND/output_upsampled_train.jsonl", lines=True).sample(frac=1, random_state=42)

df_val = pd.read_json("./HARDMATH/hardmath_output_upsampled_val.jsonl", lines=True).sample(frac=1, random_state=42)
df2_val = pd.read_json("./NEURIPS/neurips_output_upsampled_val.jsonl", lines=True).sample(frac=1, random_state=42)
df3_val = pd.read_json("./DEEPMIND/output_upsampled_val.jsonl", lines=True).sample(frac=1, random_state=42)

df_val = pd.concat([df_val, df2_val, df3_val], ignore_index=True)
df_val_0 = df_val[df_val['correct'] == 0]
df_val_sans_0 = df_val[df_val['correct'] != 0]
df_val_0 = df_val_0.sample(n=len(df_val_sans_0)//3, replace=True, random_state=42)
df_val = pd.concat([df_val_0, df_val_sans_0], ignore_index=True)

print("hardmath size: ", len(df))
print("neurips size: ", len(df2))
print("deepmind size: ", len(df3))

df = pd.concat([df, df2, df3], ignore_index=True)

print(df['correct'].value_counts())
print(df.shape)

# Embeddings
if os.path.exists("embeddings.csv"):
    X = np.loadtxt("embeddings.csv", delimiter=',')
    if len(X) != len(df):
        print("Data missing from embeddings, regenerating...")
        X = None
    X_val = np.loadtxt("embeddings_val.csv", delimiter=',')
    if len(X_val) != len(df_val):
        print("Data missing from validation embeddings, regenerating...")
        X_val = None
else:
    X = None
    X_val = None

if X is None:
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
if X_val is None:
    problems_val = df_val['problem'].tolist()
    X_val = np.empty((0, 1024), dtype=np.float32)
    for i in range(0, len(problems_val), 128):
        X_val = np.append(X_val, vo.embed(
            problems_val[i:i+128], model="voyage-3"
        ).embeddings, axis=0)
        print(f"Processed {i+128} validation problems...")
    np.savetxt("embeddings_val.csv", X_val, delimiter=',')

y = df['correct'].values
y_val = df_val['correct'].values

print(df_val['correct'].value_counts())

# Train logistic regression (multiclass, nonlinear solver)
logreg = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',  # or 'saga' for large datasets
    max_iter=1000,
    verbose=1,
    n_jobs=-1
)
logreg.fit(X, y)

# Evaluate
predictions = logreg.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix(y_val, predictions), annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
plt.savefig("logreg_confusion_matrix.png")

# Save model (pickle)
import pickle
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(logreg, f)

# Save predictions and accuracy
with open("logreg_results.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Predictions: {predictions}\n")

print("Logistic regression training complete.")
