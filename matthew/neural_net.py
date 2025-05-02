import keras
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
load_dotenv()
import voyageai
from sklearn.utils.class_weight import compute_class_weight

vo = voyageai.Client()

df = pd.read_json("./HARDMATH/hardmath_output_upsampled_train.jsonl", lines=True).sample(frac=1, random_state=42)
df2 = pd.read_json("./NEURIPS/neurips_output_upsampled_train.jsonl", lines=True).sample(frac=1, random_state=42)
df3 = pd.read_json("./DEEPMIND/output_upsampled_train.jsonl", lines=True).sample(frac=1, random_state=42)

df = pd.concat([df, df2, df3], ignore_index=True)

df_0 = df[df['correct'] == 0]
df_sans_0 = df[df['correct'] != 0]
df_0 = df_0.sample(n=len(df_sans_0)*3, replace=True, random_state=42)
df = pd.concat([df_0, df_sans_0], ignore_index=True)

df_val = pd.read_json("./HARDMATH/hardmath_output_upsampled_val.jsonl", lines=True).sample(frac=1, random_state=42)
df2_val = pd.read_json("./NEURIPS/neurips_output_upsampled_val.jsonl", lines=True).sample(frac=1, random_state=42)
df3_val = pd.read_json("./DEEPMIND/output_upsampled_val.jsonl", lines=True).sample(frac=1, random_state=42)

df_val = pd.concat([df_val, df2_val, df3_val], ignore_index=True)

df_val_0 = df_val[df_val['correct'] == 0]
df_val_sans_0 = df_val[df_val['correct'] != 0]
df_val_0 = df_val_0.sample(n=len(df_val_sans_0)//3, replace=True, random_state=42)
df_val = pd.concat([df_val_0, df_val_sans_0], ignore_index=True)

print(df_val['correct'].value_counts())

# ---------------- Training pipeline ----------------
print(df['correct'].value_counts())
print(df.shape)

# Load / regenerate embeddings
X, X_val = None, None
if os.path.exists("embeddings.csv"):
    X = np.loadtxt("embeddings.csv", delimiter=',')
    if len(X) != len(df):
        print("Training embeddings mismatch – regenerating...")
        X = None
if os.path.exists("embeddings_val.csv"):
    X_val = np.loadtxt("embeddings_val.csv", delimiter=',')
    if len(X_val) != len(df_val):
        print("Validation embeddings mismatch – regenerating...")
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
        X = np.append(X, vo.embed(problems[i:i+128], model="voyage-3").embeddings, axis=0)
        if (i+128) % 512 == 0:
            print(f"Processed {i+128} training problems...")
    np.savetxt("embeddings.csv", X, delimiter=',')

if X_val is None:
    problems_val = df_val['problem'].tolist()
    X_val = np.empty((0, 1024), dtype=np.float32)
    for i in range(0, len(problems_val), 128):
        X_val = np.append(X_val, vo.embed(problems_val[i:i+128], model="voyage-3").embeddings, axis=0)
        if (i+128) % 512 == 0:
            print(f"Processed {i+128} validation problems...")
    np.savetxt("embeddings_val.csv", X_val, delimiter=',')

# Labels
y = df['correct'].values
y_val = df_val['correct'].values

# Compute class weights to handle imbalance
classes = np.unique(y)
class_weights = dict(zip(classes, compute_class_weight(class_weight='balanced', classes=classes, y=y)))
print("Class weights:", class_weights)

# Build a slightly higher-capacity model
import keras
model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X, y, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping], class_weight=class_weights)

# Evaluation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
print("Evaluating model...")
preds = np.argmax(model.predict(X_val), axis=1)
accuracy = np.mean(preds == y_val)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_val, preds), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.tight_layout()
plt.savefig("confusion_matrix.png")

model.save('my_model.keras')

with open("results.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Model training complete.")