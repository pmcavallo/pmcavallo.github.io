---
layout: post
title: 🛡️Fraud Detection with XGBoost and scikit-learn
date: 2025-04-18
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

I'm going to generate a dataset with 10,000 records, 20 features, and 5% fraud cases to mimic the typical class imbalance in fraud detection.

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
![distribution](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/fraud1.png?raw=true) 
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

Hyperparameters are model configuration settings set before training (e.g., max_depth, learning_rate, n_estimators in XGBoost). Tuning them properly can significantly improve model performance. 
So, for instance:

- max_depth: How deep each tree can go (controls complexity)
- learning_rate: Step size when updating weights
- n_estimators: Number of trees in the model
- subsample: Fraction of training samples to use per tree

Below, I use random search to tune the XGBoost classifier for better performance. The RandomizedSearchCV tries a random subset of combinations (you choose how many with n_iter). It is much faster and usually finds near optimal parameters while still performing cross-validation to evaluate each combination.

```python
param_grid = {
    'n_estimators': randint(100, 500),       # Number of trees
    'learning_rate': uniform(0.01, 0.2),     # Smaller = slower but more accurate
    'max_depth': randint(3, 10),             # Tree depth (complexity)
    'subsample': uniform(0.5, 0.5),          # Row sampling
    'colsample_bytree': uniform(0.5, 0.5),   # Feature sampling
    'gamma': uniform(0, 1)                   # Minimum loss reduction to split
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

For each of the 20 combinations, a different model is trained using cross-validation, then scored on the validation sets using ROC AUC, and the best combination is chosen and returned as ".best_estimator_". Random sampling efficiently explore large spaces, provides robust evaluation (avoids overfitting) and chooses the best hyperparameters for a specific goal.


## ⚖️ 5. Handle Class Imbalance with SMOTE

I'm going to apply SMOTE to generate synthetic minority (fraud) class samples. In the "Airline Delay" project I dicuss SMOTE in more detail.

```python
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

xgb_best.fit(X_train_res, y_train_res)

y_proba = xgb_best.predict_proba(X_test_scaled)[:, 1]
```

---

## 🎯 6. Optimize the Classification Threshold

Instead of using 0.5 by default, we try to find the best threshold for F1 score.

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
![distribution](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/fraud2.png?raw=true) 

The Threshold Optimization plot visualizes how the model’s **Precision**, **Recall**, and **F1 Score** vary as we adjust the classification threshold.

### 🔍 What We See

- **Recall (🟠 Orange Line)**:
  - Starts high at low thresholds because most transactions are predicted as fraud.
  - Decreases as the threshold increases, meaning fewer frauds are detected.
  - High recall = fewer missed frauds, but more false positives.

- **Precision (🟢 Green Line)**:
  - Starts low, indicating many false positives.
  - Increases as the threshold rises, meaning fewer incorrect fraud labels.
  - High precision = fewer false alarms, but may miss actual frauds.

- **F1 Score (🔵 Blue Line)**:
  - A balance between Precision and Recall.
  - Peaks at an intermediate threshold (often around 0.5–0.65).
  - The optimal threshold for balanced performance is where this line is highest.

Given these results, we can:

- Choose a **lower threshold** (e.g., 0.3–0.4) if we want to **maximize recall** and detect more fraud (even with some false positives).
- Choose a **higher threshold** (e.g., 0.7–0.8) if we prefer **higher precision** and want fewer false positives.
- Choose the **threshold with highest F1 score** for **balanced performance** — this is what the code above did and often the best default choice.

This analysis helps tailor the fraud detection model to real-world business goals: whether it's minimizing risk, cost, or false positives.

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
![distribution](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/fraud3.png?raw=true) 

The confusion matrix below shows how the model performs after applying the optimized classification threshold.

|                        | Predicted Non-Fraud (0) | Predicted Fraud (1) |
|------------------------|--------------------------|----------------------|
| **Actual Non-Fraud (0)** | 1880 (✅ True Negative)       | 12 (⚠️ False Positive)      |
| **Actual Fraud (1)**     | 34 (⚠️ False Negative)        | 74 (✅ True Positive)       |

### 🧠 Explanation

- **True Negatives (1880)**: Non-fraudulent transactions correctly identified.
- **False Positives (12)**: Legitimate transactions incorrectly flagged as fraud.
- **False Negatives (34)**: Fraud cases the model missed.
- **True Positives (74)**: Fraud cases correctly detected.

### 📊 Key Performance Metrics

- **Precision** = 74 / (74 + 12) ≈ 0.86  
  → Very few false fraud alarms.

- **Recall** = 74 / (74 + 34) ≈ 0.69  
  → Catches ~69% of actual fraud.

- **F1 Score** ≈ Harmonic mean of Precision and Recall  
  → Good overall balance between detection and accuracy.

### ✅ Conclusion

The optimized threshold helps the model strike a strong balance:
- High precision to reduce false positives
- Moderate to high recall to still detect most fraud
This is an effective and practical configuration for real-world fraud detection systems.

---

## 📝 Overall Summary

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
![distribution](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/fraud4.png?raw=true) 

The SHAP summary plot visualizes the **global importance and effect** of each feature on the model's fraud predictions.

### 🔍 How to Read This Plot

- **Y-axis**: Features sorted by overall importance (top = most important).
- **X-axis (SHAP value)**: How much a feature impacts the prediction:
  - Negative SHAP value → Pushes prediction toward **non-fraud**
  - Positive SHAP value → Pushes prediction toward **fraud**

- **Color (feature value)**:
  - 🔴 Red = High feature value
  - 🔵 Blue = Low feature value

Each point represents one transaction.

### 🧠 Insights from This Plot

- `feature_12`, `feature_6`, and `feature_2` are the most influential in the model.
- High values of `feature_0` and `feature_17` often **increase fraud risk**.
- Some features, like `feature_15`, have **mixed impact** depending on the value and interaction.

### ✅ Why This Matters

This plot helps:
- Explain model predictions to non-technical stakeholders
- Identify key risk indicators
- Support transparency and fairness in model usage

### 💧 Step 4: Visualize Individual Prediction

```python
# Waterfall plot: explains a single prediction
shap.plots.waterfall(shap_values[0], max_display=10)
```
![distribution](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/fraud5.png?raw=true) 

The SHAP force plot explains how the model arrived at a specific prediction for a single transaction.

### 🧭 What This Plot Shows

- **E[f(X)] = −5.256**: This is the model's **base value** — the average predicted output across all samples.
- **f(X) = 0.26**: This is the **final prediction** for this individual case — a **low probability of fraud**.

### 🟥 Features That Increased Risk (Red)

- `Feature 19` and `Feature 9` had high positive SHAP values, **pushing the prediction toward fraud**.
- `Feature 18`, `Feature 10`, and others contributed positively as well.

### 🟦 Features That Reduced Risk (Blue)

- `Feature 11` had a large negative SHAP value (−1.62), **pulling the prediction away from fraud**.
- `Feature 6` also contributed to lowering the fraud score.

### 🧠 Interpretation

- Red features collectively added to the fraud score.
- Blue features mitigated it.
- The final prediction of **0.26** indicates a relatively **low fraud risk**, even though some features pushed in the opposite direction.

### ✅ Takeaway

This plot makes it clear **which features** drove the model’s decision and **how much they contributed**, which is essential for:
- Investigating edge cases
- Presenting evidence to stakeholders
- Ensuring accountability and fairness in model deployment

## 🤔 13. Model Discussion: XGBoost in Fraud Detection

### ✅ Advantages of Using XGBoost

- **High Predictive Accuracy**: XGBoost is one of the most powerful tree-based ensemble methods, delivering excellent results in fraud detection tasks.
- **Handles Imbalanced Data**: Works well even with skewed datasets (like 5% fraud) and allows customization through scale_pos_weight or sampling.
- **Built-in Regularization**: Reduces overfitting through L1/L2 penalties.
- **Speed and Scalability**: Efficient for large datasets due to parallel processing and optimized algorithms.
- **Interpretability**: Compatible with SHAP for detailed explanations, improving transparency in high-stakes applications like fraud.

### ❌ Disadvantages

- **Complexity**: Tuning hyperparameters (e.g., depth, learning rate, number of trees) can be time-consuming and computationally expensive.
- **Less Intuitive**: Unlike logistic regression, the decision boundaries are not easily understood by non-technical audiences.
- **Overfitting Risk**: If not properly tuned or regularized, XGBoost can overfit to training noise — especially in low-signal datasets.
- **Longer Training Time**: Compared to simpler models like logistic regression or decision trees, training XGBoost can be slower, especially with cross-validation.

---

### 🔄 Alternative Models to Consider

| Model                  | When to Use                                                                 |
|------------------------|------------------------------------------------------------------------------|
| **Logistic Regression**| For simplicity, transparency, and when features have linear relationships   |
| **Random Forest**      | For good performance with less hyperparameter tuning                        |
| **LightGBM / CatBoost**| For faster training on large data; often competitive with XGBoost            |
| **Neural Networks**    | For complex, high-volume data with nonlinear interactions                    |

---

### 📝 Summary

XGBoost is a great choice for fraud detection due to its predictive power and flexibility. However, it's important to validate its complexity against business needs. In many regulated environments, simpler models with clear explanations (like logistic regression) may be preferred — or combined with XGBoost in a champion-challenger setup.

## ⚖️ 13. Comparison: XGBoost vs Logistic Regression

To determine whether the complexity of XGBoost is justified, I'm going to compare its performance to a simpler baseline: **Logistic Regression**.

### 📦 Train Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Train logistic regression with class balancing
logreg = LogisticRegression(class_weight='balanced', random_state=42)
logreg.fit(X_train_scaled, y_train)

# Predict probabilities and class labels
y_proba_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
y_pred_logreg = (y_proba_logreg >= 0.5).astype(int)

# Evaluate performance
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

auc_logreg = roc_auc_score(y_test, y_proba_logreg)
print(f"ROC-AUC (Logistic Regression): {auc_logreg:.4f}")
```

### 🧪 What Metrics to Compare?

- If **XGBoost shows significantly better recall or F1**, it's leveraging non-linear interactions to detect fraud.
- If **Logistic Regression performs similarly**, it may be a better choice for:
  - Transparency (easy to explain to business users)
  - Simplicity (less risk of overfitting)
  - Speed and deployment (lightweight)

Logistic Regression can be a strong **baseline**. If XGBoost meaningfully improves performance, the added complexity is worthwhile. Otherwise, logistic regression may be preferable for regulated environments or when interpretability is paramount.

### 📊 Logistic Regression Results

| Metric        | Class 0 (Non-Fraud) | Class 1 (Fraud) |
|---------------|---------------------|------------------|
| Precision     | 0.99                | 0.21             |
| Recall        | 0.83                | 0.81             |
| F1 Score      | 0.90                | 0.34             |
| ROC-AUC       | -                   | **0.8832**        |

### 📊 XGBoost Results (Optimized Threshold)

| Metric        | Class 0 (Non-Fraud) | Class 1 (Fraud) |
|---------------|---------------------|------------------|
| Precision     | -                   | **0.86**         |
| Recall        | -                   | **0.69**         |
| F1 Score      | -                   | **0.76**         |
| ROC-AUC       | -                   | ~**0.91–0.92**   |

### ✅ Final Takeaways

- **Logistic Regression**:
  - Strength: **High recall** (81%) → fewer missed fraud cases
  - Weakness: **Very low precision** (21%) → many false alarms
  - Great for use cases prioritizing **detection over cost of review**

- **XGBoost**:
  - Strength: **Much higher precision and F1 score**
  - Weakness: Slightly lower recall
  - Best for **balanced detection and cost-effective fraud review**

### 🧠 Recommendation

Use:
- **Logistic Regression** when you want to catch almost every fraud (recall is critical)
- **XGBoost** when you want a **high-confidence** fraud detection system (better balance of precision, recall, and F1)

This type of side-by-side evaluation is essential for selecting the best model aligned with business objectives.
