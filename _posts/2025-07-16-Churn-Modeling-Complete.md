
# 📊 Telecom Churn Modeling & Retention Strategy 

This project simulates the use of machine learning and segmentation techniques to model customer churn, estimate revenue risk, and design retention strategies.

---

## 📁 Project Structure

1. **Modeling Pipeline**
   - Data Preparation
   - EDA + Visualization
   - Logistic Regression + Evaluation
   - XGBoost + Feature Importance
   - Scorecard Scaling

2. **Strategy & Interpretability**
   - SHAP Global + Local Explanations
   - Score Binning & CLTV Risk Simulation
   - PSI Drift Check
   - Segmentation & Profiling

---

### Step 1: Data Setup

```python
import pandas as pd
df = pd.read_excel("Telco-Customer-Churn.xlsx")

# Create RiskExposure = Monthly Charges / Tenure
df['Tenure Months'].replace(0, 1, inplace=True)
df['RiskExposure'] = df['Monthly Charges'] / df['Tenure Months']
```

---

### Step 2: Churn Rate Overview

```python
target = 'Churn Value'
df[target].value_counts(normalize=True)
```

> ✅ 26.5% churn rate

---

### Step 3: Visual EDA

- CLTV distribution by churn
- Risk Exposure vs Churn
- Correlation Heatmap
- Churn by Contract Type

*Embedded images provided in GitHub repo folder*

---

### Step 4: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

> Accuracy: 79%  
> AUC: 0.83  
> Key drivers: Contract Type, Monthly Charges, Risk Exposure

---

### Step 5: XGBoost Comparison

```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
```

- AUC: 0.83  
- Feature importance dominated by `ContractRisk`

---

### ✅ SHAP Analysis

- Global SHAP: ContractRisk and RiskExposure increase churn
- Local SHAP: TotalCharges and AutoPay reduce churn

```python
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

---

### ✅ Score Binning + CLTV Simulation

```python
df_score['Score'] = (1 - df_score['Churn_Prob']) * 600 + 300
df_score['Score'] = df_score['Score'].round()
```

> ✅ $2,009,410 in total CLTV at risk (top 3 bins)

---

### ✅ Segmentation Strategy

4-profile matrix based on churn risk & CLTV:

- 🔥 High Risk + High Value → Target for Retention
- ✅ Low Risk + Low Value → Stable group
- 💰 Low Risk + High Value → Upsell potential

---

### ✅ Scorecard Design

- Scores scaled to 300–900
- Risk bands: High, Moderate, Low
- Used to flag customers for action

---

### ✅ Monitoring with PSI

```python
from scipy.stats import entropy
psi = entropy(curr_dist, future_dist)
```

> ✅ PSI = 0.0149 → No drift

---

## 🧠 Business Impact

This project demonstrates how churn risk modeling can support:

- Targeted retention
- Revenue risk assessment
- Strategic prioritization

---
