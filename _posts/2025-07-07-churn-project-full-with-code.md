---
layout: post
title: 🔍 Customer Churn Prediction App (Streamlit + Render)
---

This project demonstrates an end-to-end machine learning pipeline to predict customer churn using simulated data. It includes data generation, preprocessing, model training, and deployment via **Streamlit** on **Render**.

---

## ✅ Step 1: Generate Simulated Churn Data (`data/generate_data.py`)

I'll simulate a Telco-style dataset with realistic churn behavior.

### 📌 Highlights:
- 1000 samples with features like tenure, charges, contract type
- Binary churn outcome (`Yes` / `No`)
- Noise-injected churn probabilities
- CSV output: `data/telco_churn.csv`

### 🧠 Concepts:
- Simulated structured data with dependencies
- Controlled randomness
- Binary classification labels

### 💻 Code
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

gender = np.random.choice(['Male', 'Female'], size=n)
senior_citizen = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
partner = np.random.choice(['Yes', 'No'], size=n)
dependents = np.random.choice(['Yes', 'No'], size=n)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n)
payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n)

tenure = np.random.randint(0, 72, size=n)
monthly_charges = np.round(np.random.normal(loc=70, scale=20, size=n), 2)
monthly_charges = np.clip(monthly_charges, 20, 130)
total_charges = tenure * monthly_charges

churn_prob = (
    0.3 * (contract == 'Month-to-month').astype(int) +
    0.2 * (monthly_charges > 80).astype(int) +
    0.1 * (senior_citizen == 1).astype(int)
)
churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1, n), 0, 1)
churn = np.where(churn_prob > 0.5, 'Yes', 'No')

df = pd.DataFrame({
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'PaymentMethod': payment_method,
    'Churn': churn
})

df.to_csv('data/telco_churn.csv', index=False)
```

---

## ✅ Step 2: Preprocessing (`model/preprocess.py`)

We use `ColumnTransformer` to encode and scale features and split data.

### 🧠 Concepts:
- Feature pipelines using scikit-learn
- Avoiding data leakage by fitting only on training data

### 💻 Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    return pd.read_csv(filepath)

def get_preprocessing_pipeline():
    categorical_features = ['gender', 'Partner', 'Dependents', 'Contract', 'PaymentMethod']
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    return ColumnTransformer([
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ])

def preprocess_and_split(df):
    X = df.drop(columns='Churn')
    y = df['Churn'].map({'No': 0, 'Yes': 1})

    preprocessor = get_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    preprocessor.fit(X_train)
    return X_train, X_test, y_train, y_test, preprocessor
```

---

## ✅ Step 3: Train & Save Model (`model/train_model.py`)

We train a `RandomForestClassifier` on the transformed data and save it.

### 💻 Code
```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import load_data, preprocess_and_split

df = load_data("data/telco_churn.csv")
X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_test_transformed)
y_proba = model.predict_proba(X_test_transformed)[:, 1]

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print(f"✅ ROC AUC Score: {auc:.4f}")

joblib.dump(model, "model/churn_model.pkl")
joblib.dump(preprocessor, "model/preprocessor.pkl")
```

---

## ✅ Step 4: Streamlit App for Render (`app/app.py`)

### 💻 Code
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

st.title("📉 Customer Churn Predictor")
st.markdown("Enter customer details below to predict the likelihood of churn.")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has a Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 20, 130, 70)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

total_charges = tenure * monthly_charges

if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment_method
    }])

    X_input = preprocessor.transform(input_df)
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    label = "🚫 Will Not Churn" if prediction == 0 else "⚠️ Will Churn"
    st.subheader(f"Prediction: {label}")
    st.write(f"Churn Probability: **{probability:.2%}**")
```

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


