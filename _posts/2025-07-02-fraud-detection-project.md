---
layout: post
title: 🛡️ Fraud Detection Using XGBoost and scikit-learn (with SMOTE and Threshold Optimization)
---
This project demonstrates a full machine learning workflow for detecting fraudulent transactions using **simulated data**, with **XGBoost**, **SMOTE** for class imbalance, **RandomizedSearchCV** for hyperparameter tuning, and **threshold optimization** to improve performance.

---

## 📦 1. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, recall_score, precision_score

from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from scipy.stats import uniform, randint

np.random.seed(42)
```

---

## 🧪 2. Simulate a Fraud Dataset

We generate a dataset with 10,000 records, 20 features, and 5% fraud cases to mimic the typical class imbalance in fraud detection.

```python
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                           n_redundant=5, weights=[0.95, 0.05],
                           flip_y=0.01, class_sep=1.0, random_state=42)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["is_fraud"] = y

sns.countplot(x="is_fraud", data=df)
plt.title("Class Distribution: Fraud vs Non-Fraud")
plt.show()
```

---

## 🧹 3. Preprocessing

Split the data into training and testing sets and apply standard scaling.

```python
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 🔍 4. Hyperparameter Tuning with RandomizedSearchCV

Use random search to tune the XGBoost classifier for better performance.

```python
param_grid = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 1)
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
print("Best Parameters:")
print(random_search.best_params_)

xgb_best = random_search.best_estimator_
```

---

## ⚖️ 5. Handle Class Imbalance with SMOTE

Apply SMOTE to generate synthetic minority (fraud) class samples.

```python
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

xgb_best.fit(X_train_res, y_train_res)

y_proba = xgb_best.predict_proba(X_test_scaled)[:, 1]
```

---

## 🎯 6. Optimize the Classification Threshold

Instead of using 0.5 by default, find the best threshold for F1 score.

```python
def evaluate_thresholds(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        scores.append((t, f1, recall, precision))

    return pd.DataFrame(scores, columns=["Threshold", "F1", "Recall", "Precision"])

threshold_results = evaluate_thresholds(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(threshold_results["Threshold"], threshold_results["F1"], label="F1 Score")
plt.plot(threshold_results["Threshold"], threshold_results["Recall"], label="Recall")
plt.plot(threshold_results["Threshold"], threshold_results["Precision"], label="Precision")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization")
plt.legend()
plt.grid(True)
plt.show()

best_thresh = threshold_results.loc[threshold_results.F1.idxmax(), "Threshold"]
print(f"Best threshold based on F1 Score: {best_thresh:.2f}")
```

---

## ✅ 7. Final Evaluation

Apply the best threshold to evaluate final performance.

```python
y_pred_opt = (y_proba >= best_thresh).astype(int)

print("Classification Report with Optimized Threshold:")
print(classification_report(y_test, y_pred_opt))

conf_matrix = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix (Optimized Threshold)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

---

## 📝 Summary

- **Simulated a highly imbalanced fraud dataset**
- **Used XGBoost** with **hyperparameter tuning**
- **Handled imbalance** with **SMOTE**
- **Improved recall and F1** through **threshold optimization**

## 🧠 8. SHAP Interpretability

SHAP (SHapley Additive exPlanations) helps explain individual predictions by assigning each feature an importance value for a given prediction.

### 🔍 Step 1: Install and Import SHAP

```python
# Install SHAP if you haven't
# !pip install shap

import shap
```

### 📊 Step 2: Initialize SHAP Explainer

```python
# Create an explainer for the trained XGBoost model
explainer = shap.Explainer(xgb_best, X_test_scaled)

# Compute SHAP values
shap_values = explainer(X_test_scaled)
```

### 🖼️ Step 3: Visualize Global Feature Importance

```python
# Summary plot: global feature importance
shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist())
```

### 💧 Step 4: Visualize Individual Prediction

```python
# Waterfall plot: explains a single prediction
shap.plots.waterfall(shap_values[0], max_display=10)
```

SHAP helps you understand:
- Which features increased the risk of fraud
- Which ones decreased it
- By how much each feature contributed

This is useful for model transparency and auditing.
