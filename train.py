from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import keras_nlp
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_json("output.jsonl", lines=True)
print(df.head(2))
print(df['correct'].value_counts())
#df['problem_correct']=df['correct'].apply(lambda x: 1 if x==True else 0)
df['problem_correct'] = df['correct'].astype(bool).astype(int)

ones_subset = df.loc[df["problem_correct"] == 0, :]
number_of_1s = len(ones_subset)

zeros_subset = df.loc[df["problem_correct"] == 1, :]
sampled_zeros = zeros_subset.sample(number_of_1s)

df = pd.concat([ones_subset, sampled_zeros], ignore_index=True)

print(df["problem_correct"].sum(), len(df))

print(df.sample(5))


X_train, X_test, y_train, y_test = train_test_split(df['problem'],df['problem_correct'], test_size=0.2)

bert_preprocess = keras_nlp.models.BertPreprocessor.from_preset("bert_base_en_uncased",trainable=True)
bert_encoder = keras_nlp.models.BertBackbone.from_preset("bert_base_en_uncased")

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

model = tf.keras.Model(inputs=[text_input], outputs = [l])

model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)

X_train = X_train.astype('string').fillna("")
y_train = y_train.fillna(0).astype(int)


model.fit(X_train.to_numpy(dtype=str), y_train.to_numpy(dtype=np.float32), epochs=10)

model.evaluate(X_test, y_test)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()

y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted

cm = confusion_matrix(y_test, y_predicted)

sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(classification_report(y_test, y_predicted))