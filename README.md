# 🫁 Pneumonia Risk Detection & Clinical Analytics System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

An AI-driven healthcare clinical decision support system combining a **Deep Neural Network (DNN)** with a **Flask web application** to assess pneumonia probability from patient vitals and laboratory biomarkers in real-time, coupled with a dataset analytics dashboard.

---

## 📌 Project Overview

Pneumonia is an acute respiratory infection that accounts for significant hospital admissions worldwide. Timely screening is critical to prevent severe complications. 

This project provides:
1. **Patient Risk Screening**: Clinicians or users can input vital signs and blood test values to receive an instant, probabilistic risk assessment with tailored medical recommendations.
2. **Batch Dataset Analytics**: Healthcare teams can upload clinical CSV datasets to analyze positive/negative prevalence, average vitals, biomarker distributions, and cohort statistics.

---

## 🧠 Machine Learning Algorithm & Model Architecture

### 1. Algorithm: Deep Neural Network (DNN) / Multi-Layer Perceptron (MLP)
The predictive model is a **Deep Feedforward Artificial Neural Network (Sequential DNN)** trained to model nonlinear relationships between vital physiological signals, inflammatory biomarkers, and pneumonia onset.

```
[10 Clinical Features]
         │
         ▼
┌─────────────────────────────────┐
│ Feature Standardization         │  (Scikit-Learn StandardScaler)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Input Layer                     │  Shape: (Batch, 10)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Hidden Layer 1 (Dense)          │  32 Neurons, ReLU Activation
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Hidden Layer 2 (Dense)          │  16 Neurons, ReLU Activation
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Output Layer (Dense)            │  1 Neuron, Sigmoid Activation
└─────────────────────────────────┘
         │
         ▼
[Pneumonia Probability: 0.0 - 1.0]
```

### 2. Clinical Input Features (10 Parameters)

| # | Feature | Clinical Description | Typical Normal Range |
|---|---|---|---|
| 1 | **Age** | Patient age in years | - |
| 2 | **Fever** | Body temperature (°C) | 36.5 – 37.5 °C |
| 3 | **Cough** | Presence of active cough | 0 (No) / 1 (Yes) |
| 4 | **Shortness of Breath** | Dyspnea indicator | 0 (No) / 1 (Yes) |
| 5 | **Heart Rate** | Resting pulse rate | 60 – 100 bpm |
| 6 | **Respiratory Rate** | Breaths per minute | 12 – 20 breaths/min |
| 7 | **SpO2** | Blood oxygen saturation | 95 – 100 % |
| 8 | **WBC Count** | White blood cell count | 4,000 – 11,000 cells/μL |
| 9 | **CRP** | C-Reactive Protein (systemic inflammation) | < 10 mg/L |
| 10 | **Procalcitonin** | Specific bacterial infection biomarker | < 0.5 ng/mL |

### 3. Model Training Specifications
- **Framework**: Keras 3 / TensorFlow 2.16
- **Feature Preprocessing**: `StandardScaler` fitted and persisted via `joblib` (`scaler.pkl`)
- **Loss Function**: `Binary Crossentropy` (optimal for probabilistic binary classification)
- **Optimizer**: `Adam` (Adaptive Moment Estimation)
- **Epochs**: 30 epochs with validation split
- **Model Format**: Serialized HDF5 (`pneumonia_model.h5`)

### 4. Clinical Risk Stratification Logic
The raw sigmoid output is converted to a risk percentage ($0\% - 100\%$) and categorized into risk tiers:

| Risk Tier | Probability Range | Alert Level | Actionable Clinical Advice |
|---|---|---|---|
| 🟢 **Low Risk** | `< 30%` | Success | Low probability. General lung health preservation, hydration, vaccines, and monitor for changes. |
| 🟡 **Moderate Risk** | `30% – 69.9%` | Warning | Symptoms consistent with potential respiratory illness. Medical review and diagnostic chest X-ray recommended. |
| 🔴 **High Risk** | `≥ 70%` | Danger | High probability of acute infection. Immediate physician consultation or urgent care required. |

---

## ✨ Key Features Present in the Project

- **Interactive Clinical Form**: Clean, accessible form equipped with real-time field validation and normal-range clinical guidelines.
- **Dynamic Visual Risk Meter**: Visual gauge displaying computed percentage, risk category, and structured medical advice.
- **Dataset Analytics Explorer (`/analytics`)**: Upload any CSV dataset to instantly inspect:
  - Total patient volume
  - Pneumonia positive vs negative counts & percentage
  - Average patient age stratified by pneumonia status
  - Cohort vital sign averages (Fever, SpO2, Heart rate)
- **High-Performance REST API**: JSON-based asynchronous endpoints:
  - `POST /predict`: Real-time single-patient prediction
  - `POST /upload_dataset`: In-memory multipart CSV statistical analysis
  - `GET /health`: Health-check endpoint for cloud uptime monitoring
- **Cloud-Optimized Architecture**: Configured with `gunicorn`, `tensorflow-cpu`, and constrained worker threads to run smoothly on free-tier 512 MB memory environments.

---

## 🛠️ Technology Stack & Libraries Used

### Machine Learning & Data Processing
- **[TensorFlow CPU](https://www.tensorflow.org/) (`tensorflow-cpu`)**: Lightweight neural network runtime without CUDA overhead.
- **[Keras 3](https://keras.io/)**: High-level deep learning API for model definition and inference.
- **[Scikit-Learn](https://scikit-learn.org/)**: Standard scaling and normalization of numerical inputs.
- **[Pandas](https://pandas.pydata.org/)**: Dataset parsing, aggregation, and cohort analytics.
- **[NumPy](https://numpy.org/)**: Array vectorization and mathematical operations (`numpy<2.0.0` for TF C-API compatibility).
- **[Joblib](https://joblib.readthedocs.io/)**: Fast object serialization for the trained scaler.

### Web Backend & Production Serving
- **[Flask](https://flask.palletsprojects.com/)**: Python micro-framework handling routing, templating, and REST endpoints.
- **[Gunicorn](https://gunicorn.org/)**: Production WSGI HTTP server configured with multi-threading and timeout handling.

### Frontend
- **HTML5 & CSS3**: Modern responsive UI with gradients, cards, and smooth transitions.
- **Vanilla JavaScript**: Asynchronous `fetch` API for zero-refresh interactions.

---

## 📂 Project Structure

```plaintext
dnn1/
├── app.py                     # Flask application & inference routes (/predict, /upload_dataset, /health)
├── train_model.py             # Model training script (DNN architecture & scaler pipeline)
├── pneumonia_model.h5         # Pre-trained deep neural network model
├── scaler.pkl                 # Fitted StandardScaler for 10 input features
├── requirements.txt           # Production Python dependencies
├── render.yaml                # Render Blueprint deployment definition
├── runtime.txt                # Python runtime specification (python-3.11.9)
├── Procfile                   # Gunicorn WSGI start command
├── templates/
│   ├── index.html             # Patient risk screening web interface
│   └── analytics.html         # Dataset batch analytics web interface
├── uploads/
│   └── pneumonia_lab_dataset_1000.csv  # Sample testing dataset (1,000 patient records)
└── README.md                  # Comprehensive project documentation
```

---

## 💻 Local Setup & Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/N-ABHINAND/PNEUMONIA-DETECTION-USING-DNN.git
   cd PNEUMONIA-DETECTION-USING-DNN
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

