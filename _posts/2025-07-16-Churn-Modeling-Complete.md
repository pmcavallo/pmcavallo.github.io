
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

This step explores visual patterns in the dataset to uncover variables strongly associated with churn behavior. We focus on Customer Lifetime Value (CLTV), Risk Exposure, Contract Type, and pairwise correlations.

---

#### 🔹 1. CLTV Distribution by Churn Status

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.kdeplot(data=df, x='CLTV', hue='Churn Value', fill=True, common_norm=False,
            palette="Set2", alpha=0.5, linewidth=1.5)
plt.title("Distribution of CLTV by Churn Status")
plt.xlabel("Customer Lifetime Value")
plt.ylabel("Density")
plt.legend(title='Churn', labels=["Retained", "Churned"])
plt.tight_layout()
plt.show()
```
Interpretation:
Churned customers tend to have lower Customer Lifetime Value (CLTV), while retained customers peak around $5,000–$6,000. This confirms that CLTV can serve as an important predictor of long-term customer retention.

---

### 🔹 2. Risk Exposure vs Churn (Boxplot)

```python
df['RiskExposure'] = df['Monthly Charges'] / df['Tenure Months'].replace(0, 1)

plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='Churn Value', y='RiskExposure')
plt.xticks([0, 1], ['No Churn', 'Churned'])
plt.title("Risk Exposure vs Churn Value")
plt.ylabel("Risk Exposure (Monthly Charges / Tenure)")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.tight_layout()
plt.show()
```

Interpretation:
Churned customers exhibit significantly higher Risk Exposure, which is calculated as monthly charges divided by tenure. This metric captures financial volatility and early disengagement — both of which are risk flags.

---

### 🔹 3. Correlation Heatmap

```python
numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges',
                  'CLTV', 'Churn Value', 'RiskExposure']

plt.figure(figsize=(8,6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()
```

Interpretation:
- Churn Value is negatively correlated with Tenure Months (-0.35) and positively correlated with RiskExposure (+0.42)
- CLTV is positively associated with both Tenure and Total Charges
- Monthly Charges have weak correlation with churn but are important when combined with tenure (i.e., RiskExposure)

---

### 🔹 4. Churn Rate by Contract Type

```python
plt.figure(figsize=(7,5))
sns.barplot(data=df, x='Contract', y='Churn Value', ci='sd')
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.xlabel("Contract")
plt.tight_layout()
plt.show()
```

Interpretation:
Month-to-month contracts show a churn rate exceeding 40%, far higher than one- and two-year contracts. Contract type is a powerful signal of customer loyalty and retention behavior.

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
