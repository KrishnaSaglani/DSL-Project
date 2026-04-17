import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("dataset.csv")

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values

# Normalize
X = X / 1000.0

X = X.reshape(-1,5,1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(8, 2, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=50)

model.save("model.h5")

print("Model trained")