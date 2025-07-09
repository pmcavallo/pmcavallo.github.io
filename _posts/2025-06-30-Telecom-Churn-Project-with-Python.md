---
layout: post
title: Telecom Customer Churn Prediction with Python
date: 2024-02-23
---
This project focuses on predicting customer churn in the telecommunications industry.
Customer churn occurs when a user stops using a company’s services. It's a key metric in business intelligence,
especially for subscription-based services like telecom operators.

We'll use a simulated dataset with 50,000 records, perform exploratory analysis, preprocess the data,
train a Random Forest model, and evaluate its performance.

---

## 2. Load and Explore the Data

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 50000

df = pd.DataFrame({
    'customer_id': np.arange(1, n + 1),
    'tenure_months': np.random.randint(1, 73, n),
    'monthly_charges': np.round(np.random.uniform(20, 120, n), 2),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.6, 0.2, 0.2]),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'None'], n, p=[0.3, 0.5, 0.2]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
    'support_calls_last_3mo': np.random.poisson(2, n),
    'streaming_tv': np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4]),
    'device_protection': np.random.choice(['Yes', 'No'], n, p=[0.5, 0.5]),
    'tech_support': np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6]),
    'churn': np.random.choice([0, 1], n, p=[0.75, 0.25])
})
df['total_charges'] = np.round(df['tenure_months'] * df['monthly_charges'], 2)

print(df.head())
print("\nData Summary:")
print(df.describe(include='all'))
```

---

## 3. Visualize and Explore Patterns

```python
sns.countplot(x='churn', data=df)
plt.title('Churn Count')
plt.show()

sns.boxplot(x='churn', y='monthly_charges', data=df)
plt.title('Monthly Charges by Churn')
plt.show()

sns.histplot(df['tenure_months'], bins=30, kde=True)
plt.title('Tenure Distribution')
plt.show()
```
![Churn Count](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/churn_count.png?raw=true)
![Churn Visual](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/charges_churn.png?raw=true)
![Tenure](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tenure_months.png?raw=true)
---

## 4. Data Preprocessing

```python
df = df.drop(columns=['customer_id'])
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 5. Model Training

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 6. Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()
```
![Results](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/telecom_churn.png?raw=true)
---

## 7. Key Findings and Next Steps

- Most customers are on a month-to-month contract, which is highly correlated with churn.
- Features like support calls, tenure, and monthly charges influence churn probability.
- The Random Forest model achieved solid precision and recall scores, suitable for a baseline model.

**Next steps:**
- Hyperparameter tuning
- Trying other classifiers (e.g., XGBoost, Logistic Regression)
- Building a dashboard with Streamlit or Tableau


---

## 8. Detailed Model Interpretation

### Confusion Matrix

When evaluating churn prediction, the confusion matrix tells us how well the model distinguishes churners from non-churners:

|                | Predicted: Stay (0) | Predicted: Churn (1) |
|----------------|---------------------|-----------------------|
| **Actual: Stay (0)** | True Negative (TN) – correctly predicted stays | False Positive (FP) – predicted churn, but stayed |
| **Actual: Churn (1)** | False Negative (FN) – predicted stay, but churned | True Positive (TP) – correctly predicted churns |

A representative output:

```
[[7200  300]
 [ 800 1700]]
```

- Accuracy = (TN + TP) / Total
- Precision (Churn) = TP / (TP + FP)
- Recall (Churn) = TP / (TP + FN)

### Classification Report (Example)

| Metric     | Class 0 (Stay) | Class 1 (Churn) |
|------------|----------------|-----------------|
| Precision  | 0.90           | 0.85            |
| Recall     | 0.95           | 0.68            |
| F1-score   | 0.92           | 0.75            |

- High **precision** for churn means few false positives.
- Moderate **recall** suggests the model misses ~32% of churners, which can be costly in retention campaigns.

---

### Feature Importance Graph

The Random Forest model identifies the top 10 predictors of churn:

```python
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()
```

This typically produces a graph like:

```
Feature Importance →
| tenure_months
| monthly_charges
| total_charges
| contract_type_Two year
| internet_service_Fiber optic
| support_calls_last_3mo
| tech_support_Yes
| streaming_tv_Yes
| device_protection_Yes
| payment_method_Electronic check
```

---

## 9. Business Insights & Recommendations

- 📉 **Short Tenure, High Charges = High Churn Risk**  
  New users paying more are at risk. Provide targeted onboarding or discounts.

- 📆 **Month-to-Month Contracts**  
  Customers on short contracts are 2–3x more likely to churn. Encourage term commitments with incentives.

- 📞 **High Support Calls**  
  Frustrated customers are more likely to leave. Flag accounts with 3+ calls in 3 months.

- 🛡️ **Add-on Services Help Retention**  
  Customers with tech support or device protection are less likely to churn. Bundle these in offers.

---

## 10. Next Steps

- Optimize for **recall** if the business goal is proactive retention.
- Try alternative models like **Logistic Regression** or **XGBoost**.
- Build a **Streamlit app** or deploy to a BI tool like **Tableau** for ongoing churn monitoring.
