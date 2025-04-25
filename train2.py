import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load your dataset into a Pandas DataFrame
# Example: IMDb review dataset (sampled)
# Replace this with pd.read_csv("your_file.csv")
rawdf = pd.read_json("output.jsonl", lines=True)

df = pd.DataFrame()
df["text"] = rawdf["problem"]
df['label'] = rawdf['correct'].astype(bool).astype(int)

ones_subset = df.loc[df["label"] == 0, :]
number_of_1s = len(ones_subset)

zeros_subset = df.loc[df["label"] == 1, :]
sampled_zeros = zeros_subset.sample(number_of_1s)

df = pd.concat([ones_subset, sampled_zeros], ignore_index=True)

print(df["label"].sum(), len(df))
print(df.sample(5))


# Ensure binary labels: 0 (negative), 1 (positive)

# 2. Split into train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)


# 3. Tokenize
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

train_encodings = tokenizer(
    list(train_texts), truncation=True, padding=True, max_length=256, return_tensors="tf"
)
test_encodings = tokenizer(
    list(test_texts), truncation=True, padding=True, max_length=256, return_tensors="tf"
)

# 4. Prepare tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels.values
)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels.values
)).batch(16)

# 5. Load and compile BERT model
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy()]
)

# 6. Train the model
model.save_weights("modelsave")
model.fit(train_dataset, epochs=2, steps_per_epoch=10, validation_data=test_dataset)
model.save_weights("modelsave")

# 7. Evaluate
logits = model.predict(test_dataset).logits
preds = np.argmax(logits, axis=1)

accuracy = accuracy_score(test_labels, preds)
print(f"Test Accuracy: {accuracy:.4f}")
