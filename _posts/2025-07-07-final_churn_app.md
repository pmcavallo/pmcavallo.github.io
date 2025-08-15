---
layout: post
title:  Customer Churn Prediction App (Deployed on Render)
---

This project is an end-to-end machine learning web app for predicting customer churn using a trained classification model.

🔗 **Live App:** [https://churn-prediction-app-dxft.onrender.com](https://churn-prediction-app-dxft.onrender.com)

---

## App Preview

![Churn Prediction App Screenshot](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/streamlit_ui.png?raw=true)

---

## Project Overview

This app:
- Trains a `RandomForestClassifier` to predict churn
- Encodes/preprocesses input features using `ColumnTransformer`
- Uses `joblib` to save/load model artifacts
- Provides a user-friendly interface using `Streamlit`
- Is deployed serverlessly using **Render**

---
## ✅ Step 1: Generate Simulated Churn Data (`data/generate_data.py`)

I'll simulate a telecom dataset with realistic churn behavior.

### Highlights:
- 1000 samples with features like tenure, charges, contract type
- Binary churn outcome (`Yes` / `No`)
- Noise-injected churn probabilities
- CSV output: `data/telco_churn.csv`

### Concepts:
- Simulated structured data with dependencies
- Controlled randomness
- Binary classification labels

### Code
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

# Quick sanity check on churn distribution
print("Churn distribution:", df['Churn'].value_counts(normalize=True).round(2))

```

---

## Model Training and Preprocessing (Python)

### 1. Load and Preprocess Data

### Concepts:
- Feature pipelines using scikit-learn
- Avoiding data leakage by fitting only on training data

### Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_and_split(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    categorical = X.select_dtypes(include='object').columns.tolist()
    numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    X_transformed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor
```

---

### 2. Train and Save the Model

I'll train a `RandomForestClassifier` on the transformed data and save it.

### Code

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model(X_train, y_train, preprocessor):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/churn_model.pkl')
    joblib.dump(preprocessor, 'model/preprocessor.pkl')

    return model
```
---
- The target `Churn` is binary encoded.
- Final model and preprocessor are saved as `.pkl` files for use in the web app.


### 3. Execute Training Script

```python
if __name__ == "__main__":
    df = load_data("data/telco_churn.csv")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    train_and_save_model(X_train, y_train, preprocessor)
```

---

## 🧾 Decision Notes

- I chose a **RandomForestClassifier** for its interpretability and robustness on synthetic churn data. **Streamlit** was selected for its speed in building prototypes, and **Render** for seamless, cost-effective cloud deployment without complex infrastructure.
 

## 4: Streamlit App for Render 

### Code
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

## ⚙️ Local Setup

### ✅ 1. Clone the Repo

```bash
git clone https://github.com/pmcavallo/churn-prediction-app.git
cd churn-prediction-app
```

### ✅ 2. Install Dependencies

```bash
pip install -r requirements.txt
```
## 📄 Requirements (`requirements.txt`)

```
streamlit
pandas
numpy
scikit-learn
joblib
```

### ✅ 3. Train the Model 

```bash
python model/train_model.py
```

This saves:
- `model/churn_model.pkl`
- `model/preprocessor.pkl`

### ✅ 4. Launch the Streamlit App

```bash
streamlit run app/app.py
```

---

## 🌐 Deploying to Render

Render is a free serverless platform that supports Python + Streamlit.

### ✅ Setup Steps

1. Push all files to my GitHub repo
2. Go to [https://render.com](https://render.com)
3. Click **"New Web Service"**
4. Connect the GitHub repo
5. Configure the following:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/app.py --server.port $PORT`
6. Done! 🎉 The app is live.

📌 Add a `render.yaml` like this to automate config:

```yaml
services:
  - type: web
    name: churn-prediction-app
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/app.py --server.port $PORT
    autoDeploy: true
```

---

**Considered Alternatives:**  
- **Logistic Regression**: simpler but sacrificed predictive accuracy.  
- **Flask/Dash**: more boilerplate; Streamlit offered quicker iteration.  
- **AWS/Heroku**: suitable but rendered deployment heavier; Render provided instant streaming of updates.


## Tech Stack

| Purpose         | Tool            |
|-----------------|-----------------|
| Language        | Python 3        |
| ML Library      | scikit-learn    |
| Web UI          | Streamlit       |
| Deployment      | Render          |
| Model Storage   | joblib          |
| Dataset         | Simulated Telco Churn |

---

## Author

**Paulo Cavallo**  
🔗 [LinkedIn](https://www.linkedin.com/in/paulo-cavallo/)  
🧠 [GitHub](https://github.com/pmcavallo)

---

## 📄 License

This project is available under the MIT License.
---



