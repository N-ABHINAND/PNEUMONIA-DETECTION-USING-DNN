# Pneumonia Risk Detection & Analytics System

An AI-powered web application built with Flask, TensorFlow/Keras, and scikit-learn to screen patient risk for pneumonia based on vital signs and lab markers, along with dataset analytics.

---

## 🚀 Deploying to Render (Step-by-Step)

Deploying this application to [Render](https://render.com) is free and takes just a few minutes.

### Step 1: Push your project to GitHub

1. Open your terminal in this project directory:
   ```bash
   git add .
   git commit -m "Configure project for Render deployment"
   ```
2. Create a new repository on [GitHub](https://github.com/new) (e.g. `pneumonia-detection`).
3. Link and push your code:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/pneumonia-detection.git
   git push -u origin main
   ```

---

### Step 2: Deploy on Render

You can deploy using either **Option A (Render Blueprint - Fastest)** or **Option B (Manual Web Service)**.

#### Option A: 1-Click Blueprint (Recommended)
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** > **Blueprint**.
3. Connect your GitHub repository.
4. Render will automatically detect [`render.yaml`](render.yaml) and configure:
   - **Service Name**: `pneumonia-risk-app`
   - **Environment**: Python 3.11.9
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app`
5. Click **Apply**. Render will automatically build and deploy!

#### Option B: Manual Web Service
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** > **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your repository.
4. Fill in the following settings:
   - **Name**: `pneumonia-risk-app` (or any name you prefer)
   - **Region**: Choose the closest region (e.g., Singapore, Oregon, Frankfurt)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app
     ```
   - **Instance Type**: `Free`
5. In **Advanced Settings**:
   - Add Environment Variable:
     - `PYTHON_VERSION` = `3.11.9`
   - Health Check Path: `/health`
6. Click **Create Web Service**.

---

## ⚙️ Why These Settings Were Configured

1. **`tensorflow-cpu` vs full `tensorflow`**:
   - Render's Free tier provides 512 MB RAM and no GPU.
   - Standard TensorFlow downloads heavy CUDA libraries and can cause build timeouts or Out-Of-Memory (OOM) crashes.
   - `tensorflow-cpu` is 10x smaller, installs fast, and runs identically for inference.
2. **`numpy<2.0.0`**:
   - TensorFlow 2.16 C-extensions require NumPy 1.x compatibility. Specifying `numpy<2.0.0` avoids `ImportError: numpy.core.umath failed to import`.
3. **`--workers 1 --threads 2` in Gunicorn**:
   - Loading deep learning models in multiple worker processes duplicates memory usage.
   - 1 worker with 2 threads loads the model once and handles concurrent requests smoothly within 512MB RAM.
4. **`--timeout 120`**:
   - Prevents worker timeout during model loading on initial spin-up.

---

## 💡 Render Free Tier Notes
- **Spindown after inactivity**: Render Free Tier services spin down after 15 minutes of inactivity. When a new request arrives, it may take 30–50 seconds to wake up (cold start).
- Once spun up, requests are processed immediately.
