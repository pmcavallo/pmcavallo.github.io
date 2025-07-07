---
layout: post
title: 🔍 Customer Churn Prediction App (Streamlit + Render)
---
This project demonstrates an end-to-end machine learning pipeline to predict customer churn using simulated data. It includes data generation, preprocessing, model training, and deployment via **Streamlit** on **Render**.

---

## ✅ Step 1: Generate Simulated Churn Data (`data/generate_data.py`)

We simulate a Telco-style dataset with realistic churn behavior.

### 📌 Highlights:
- 1000 samples with features like tenure, charges, contract type
- Binary churn outcome (`Yes` / `No`)
- Noise-injected churn probabilities
- CSV output: `data/telco_churn.csv`

### 🧠 Concepts:
- Simulated structured data with dependencies
- Controlled randomness
- Binary classification labels

---

## ✅ Step 2: Preprocessing (`model/preprocess.py`)

We use `ColumnTransformer` to encode and scale features and split data.

### 📌 Highlights:
- One-hot encoding for categorical variables
- Standard scaling for numeric features
- 80/20 stratified train-test split

### 🧠 Concepts:
- Feature pipelines using scikit-learn
- Reusable preprocessor for deployment
- Avoiding data leakage by fitting only on training data

---

## ✅ Step 3: Train & Save Model (`model/train_model.py`)

We train a `RandomForestClassifier` on the transformed data.

### 📌 Highlights:
- Model evaluation via AUC and classification report
- `joblib` saves model and preprocessor
- Output:
  - `model/churn_model.pkl`
  - `model/preprocessor.pkl`

### 🧠 Concepts:
- Supervised learning with scikit-learn
- ROC AUC interpretation
- Model persistence for production

---

## ✅ Step 4: Streamlit App for Render (`app/app.py`)

This app lets users enter customer info and view churn predictions.

### 📌 Highlights:
- Streamlit UI with input widgets
- On form submission:
  - Preprocess user input
  - Make prediction and show churn probability

### 🧠 Concepts:
- Streamlit interactivity (`selectbox`, `slider`, `button`)
- Model inference with saved pipeline
- Real-time binary classification

---

## 📦 File Structure

```
churn-prediction-app/
├── app/
│   └── app.py
├── data/
│   └── telco_churn.csv
├── model/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── churn_model.pkl
│   └── preprocessor.pkl
├── requirements.txt
├── render.yaml (optional)
└── README.md
```

---

## 🚀 Deploying to Render

1. Push the full project to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Use:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `streamlit run app/app.py --server.port $PORT`
4. You’re live!

---

## 📄 Requirements (`requirements.txt`)

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 🧠 Final Thoughts

This project showcases the full ML lifecycle:
- Simulate → Train → Evaluate → Deploy
- Modular codebase with explainable steps
- Production-ready Streamlit interface

Use it as a portfolio piece, a template for your next project, or a teaching tool!

