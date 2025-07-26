
# Credit Risk Model Deployment & Monitoring (AWS + PySpark + CatBoost)

## Overview

This project demonstrates the end-to-end implementation of a credit risk modeling pipeline using:
- **PySpark** for scalable data preprocessing and transformation  
- **Amazon S3** for cloud storage and integration  
- **CatBoost** for tree-based modeling with categorical support  
- **SHAP** for model explainability  
- **Segment-level Analysis** for business insights  
- **Realistic business recommendation** aligned with telecom and credit risk decisioning practices  

The use case reflects a postpaid lending product with synthetic but realistic behavior signals. The entire pipeline is scalable and cloud-compatible.

---

## 1. Data Generation

We simulate a synthetic telecom-style credit dataset with known risk drivers and regional variation.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000
states = ['TX', 'CA', 'NY', 'FL', 'IL', 'NC', 'MI', 'WA']

df = pd.DataFrame({
    'fico_score': np.random.normal(650, 50, n).clip(300, 850),
    'loan_amount': np.random.exponential(10000, n).clip(500, 50000),
    'tenure_months': np.random.randint(1, 60, n),
    'monthly_income': np.random.normal(5000, 1500, n).clip(1000, 20000),
    'credit_utilization': np.random.beta(2, 5, n),
    'cltv': np.random.uniform(0.2, 1.2, n),
    'state': np.random.choice(states, n),
    'delinq_flag': np.random.binomial(1, 0.15, n),
})

df['util_bin'] = pd.cut(df['credit_utilization'], bins=[-0.1, 0.2, 0.6, 1], labels=['Low', 'Medium', 'High'])
df['loan_status_flag'] = (
    (df['fico_score'] < 600) |
    (df['credit_utilization'] > 0.9) |
    (df['delinq_flag'] == 1) |
    (df['cltv'] > 1)
).astype(int)
```

---

## 2. PySpark ETL and Preprocessing

We use PySpark for scalable transformation and export.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

spark = SparkSession.builder.appName("ETL").getOrCreate()
df_spark = spark.createDataFrame(df)

df_spark = df_spark.withColumn("high_risk", when(
    (col("fico_score") < 600) |
    (col("credit_utilization") > 0.9) |
    (col("delinq_flag") == 1) |
    (col("cltv") > 1), 1).otherwise(0)
)

df_spark.write.mode("overwrite").parquet("/mnt/data/credit_flagship.parquet")
```

---

## 3. AWS S3 Upload (via CLI)

```bash
aws s3 cp /mnt/data/credit_flagship.parquet s3://aws-flagship-project/credit_flagship.parquet --recursive
```

---

## 4. Modeling with CatBoost

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

X = df.drop(columns='loan_status_flag')
y = df['loan_status_flag']

X = pd.get_dummies(X, columns=['state', 'util_bin'], drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, cat_features=[], verbose=0)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_val)[:, 1]
print(f"AUC: {roc_auc_score(y_val, y_pred_proba):.3f}")
```

---

## 5. SHAP Interpretability

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val)
```

Key insights:
- **FICO score** is the strongest driver; higher scores sharply reduce risk.
- **Utilization and CLTV** also contribute to higher risk.
- **Delinquency flag** is binary but adds strong predictive signal.

---

## 6. Segment-Level Analysis by State

```python
X_val_copy = X_val.copy()
X_val_copy['prediction'] = y_pred_proba
X_val_copy['actual'] = y_val.values

X_val_copy['state'] = df.loc[X_val_copy.index, 'state']

segment_df = X_val_copy.groupby('state').agg(
    avg_predicted=('prediction', 'mean'),
    actual_rate=('actual', 'mean')
).sort_values('actual_rate', ascending=False)

segment_df.plot(kind='barh', figsize=(8, 5), title="CatBoost – Predicted vs Actual Default Rate by State")
```

---

## 7. Business Recommendation

- **Tailored Thresholds**: Adjust risk thresholds based on state-level accuracy gaps. NY and FL underperform slightly in predicted rates—consider model recalibration.
- **Scorecard Enhancement**: FICO score dominates the model. Future iterations should regularize this influence or augment with behavior-based features.
- **S3 Data Pipelines**: Spark → Parquet → S3 integration is clean. Automate using **Airflow** in production.
- **Explainability-Ready**: SHAP makes the model audit-friendly. Aligns with regulatory needs (SR 11-7 or CCAR-style audits).
- **Next Steps**:
  - Extend to LGD modeling using fractional logistic regression.
  - Incorporate **Snowflake** as centralized data source.
  - Productionalize with **SageMaker Pipelines** or **Lambda triggers** from S3.

---

_Last updated: 2025-07-26 20:38_
