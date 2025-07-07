
# 📊 Churn Prediction Model Deployment (Simulated Data)

This project demonstrates an end-to-end machine learning pipeline for customer churn prediction using **simulated data**, with full deployment via **Streamlit** and **Render**. It is modular, well-documented, and designed for production-style workflow.

---

## ✅ Step 1: Generate Simulated Data (`generate_data.py`)

We begin by simulating a Telco-style churn dataset with realistic patterns based on contract type, charges, tenure, and demographics.

### 🎯 Objective
To simulate a labeled dataset for a binary classification problem — predicting **churn** (`Yes` or `No`) — using a mix of numeric and categorical customer attributes.

### 🧠 Key Concepts

#### ✅ Synthetic Data Generation
We generate data using `numpy` and `pandas`, which gives us full control over:
- Feature distributions
- Target relationships
- Sample size and variability

#### 🎯 Binary Target: `Churn`
We model `Churn` as `Yes` or `No` by simulating a probability based on customer traits, then apply a threshold (0.5).

#### 🧩 Feature Engineering in Simulation
- `tenure`: Number of months the customer has stayed
- `MonthlyCharges`, `TotalCharges`: Billing information
- `Contract`: Strongly predictive of churn
- `SeniorCitizen`: Impacts probability of churn

#### 📊 Target Relationship
Churn is more likely if:
- `Contract` is month-to-month
- `MonthlyCharges` are high
- `SeniorCitizen` is 1 (True)

We simulate this behavior using conditional logic and add **Gaussian noise** to keep it realistic.

#### 📁 Output
- File: `data/telco_churn.csv`
- Rows: 1000
- Features: 10 + 1 binary target (`Churn`)

### 🧪 Code: `data/generate_data.py`
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Categorical features
gender = np.random.choice(['Male', 'Female'], size=n)
senior_citizen = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
partner = np.random.choice(['Yes', 'No'], size=n)
dependents = np.random.choice(['Yes', 'No'], size=n)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n)
payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n)

# Numeric features
tenure = np.random.randint(0, 72, size=n)
monthly_charges = np.round(np.random.normal(loc=70, scale=20, size=n), 2)
monthly_charges = np.clip(monthly_charges, 20, 130)
total_charges = tenure * monthly_charges

# Simulate churn probability
churn_prob = (
    0.3 * (contract == 'Month-to-month').astype(int) +
    0.2 * (monthly_charges > 80).astype(int) +
    0.1 * (senior_citizen == 1).astype(int)
)
churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1, n), 0, 1)
churn = np.where(churn_prob > 0.5, 'Yes', 'No')

# Create DataFrame
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

# Save to CSV
df.to_csv('data/telco_churn.csv', index=False)
print("✅ Simulated churn dataset saved to 'data/telco_churn.csv'")
```

---

✅ **Next Step**: Build `preprocess.py` to encode categorical variables, scale numerical values, and split into train/test sets.
