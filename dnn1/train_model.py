import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# 1) Load dataset
df = pd.read_csv("sample dataset/pneumonia_lab_dataset_1000.csv")

# 2) Features and label: use all clinically important fields
X = df[[
    "Age",
    "Fever",
    "Cough",
    "Shortness_of_breath",
    "Heartrate",
    "Respiratory_rate",
    "SpO2",
    "WBC",
    "CRP",
    "Procalcitonin"
]].values
y = df["Pneumonia"].values

# 3) Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Build and train DNN
model = keras.Sequential([
    keras.layers.Input(shape=(X_scaled.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_scaled, y, epochs=30, batch_size=32, validation_split=0.2)

# 5) Save scaler and model
joblib.dump(scaler, "scaler.pkl")
model.save("pneumonia_model.h5")
