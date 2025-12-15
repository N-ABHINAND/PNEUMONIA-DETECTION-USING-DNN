from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

app = Flask(__name__)

# ---------- LOAD MODEL + SCALER ----------
scaler = joblib.load("scaler.pkl")
model = tf.keras.models.load_model("pneumonia_model.h5")


# ---------- PAGES ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


# ---------- PATIENT PREDICTION ----------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = [
            float(data["age"]),
            float(data["fever"]),
            float(data["cough"]),
            float(data["shortness_of_breath"]),
            float(data["heart_rate"]),
            float(data["respiratory_rate"]),
            float(data["spo2"]),
            float(data["wbc"]),
            float(data["crp"]),
            float(data["procalcitonin"]),
        ]
    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Missing input field: {str(e)}"
        })

    X = np.array(features).reshape(1, -1)

    # model prediction
    X_scaled = scaler.transform(X)
    prob = float(model.predict(X_scaled)[0][0])
    risk_probability = round(prob * 100, 2)

    # map probability → level, color, advice
    if risk_probability < 30:
        risk_level = "Low Risk"
        color = "success"
        advice = (
            "Current results suggest a low risk of pneumonia. "
            "Keep looking after your lungs: avoid smoking, stay active, drink enough fluids, "
            "and keep vaccinations such as flu and pneumonia shots up to date. "
            "If you later develop high fever, chest pain, trouble breathing, or a cough that does not improve, "
            "arrange a check-up with a doctor."
        )
    elif risk_probability < 70:
        risk_level = "Moderate Risk"
        color = "warning"
        advice = (
            "There are some signs that may be consistent with pneumonia. "
            "A medical review in the near term is recommended so a doctor can examine you and decide on tests "
            "such as a chest X‑ray or blood work. "
            "Seek urgent care sooner if your breathing becomes harder, fever stays above 39 °C, "
            "or your cough produces thick yellow/green or bloody mucus."
        )
    else:
        risk_level = "High Risk"
        color = "danger"
        advice = (
            "Your results suggest a high risk of pneumonia or another serious lung infection. "
            "Do not rely only on this tool—contact a doctor or emergency service as soon as possible, "
            "especially if you have chest pain, very fast breathing, bluish lips or fingers, confusion, "
            "or feel severely unwell. "
            "Rapid medical assessment is important for proper diagnosis and treatment."
        )

    result = {
        "success": True,
        "risk_probability": risk_probability,
        "risk_level": risk_level,
        "confidence": "Model connected",
        "advice": advice,
        "color": color
    }
    return jsonify(result)


# ---------- DATASET ANALYTICS ----------
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    file = request.files.get("file")
    if not file:
        return jsonify({"success": False, "error": "No file uploaded"})

    try:
        df = pd.read_csv(file)

        pneumonia_col = "Pneumonia"
        age_col = "Age"
        fever_col = "Fever"
        spo2_col = "SpO2"
        heart_col = "Heartrate"

        total_patients = len(df)
        pneumonia_positive = int((df[pneumonia_col] == 1).sum())
        pneumonia_negative = int((df[pneumonia_col] == 0).sum())
        percentage_positive = round(pneumonia_positive * 100 / total_patients, 2)

        data_out = {
            "success": True,
            "total_patients": total_patients,
            "pneumonia_positive": pneumonia_positive,
            "pneumonia_negative": pneumonia_negative,
            "percentage_positive": percentage_positive,
            "avg_age": round(df[age_col].mean(), 2),
            "age_pneumonia": round(df[df[pneumonia_col] == 1][age_col].mean(), 2),
            "age_no_pneumonia": round(df[df[pneumonia_col] == 0][age_col].mean(), 2),
            "avg_fever": round(df[fever_col].mean(), 2),
            "avg_spo2": round(df[spo2_col].mean(), 2),
            "avg_heart_rate": round(df[heart_col].mean(), 2),
        }
        return jsonify(data_out)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
