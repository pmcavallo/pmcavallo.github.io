
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

![cltv by churn](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile4.png?raw=true) 

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
![risk vs churn](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile3.png?raw=true) 

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
![heatmap](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile2.png?raw=true) 

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
![churn by contract](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile.png?raw=true) 

Interpretation:
Month-to-month contracts show a churn rate exceeding 40%, far higher than one- and two-year contracts. Contract type is a powerful signal of customer loyalty and retention behavior.

---

### Step 4: Logistic Regression

I begin with a baseline classification model to predict customer churn using logistic regression. The features used include a mix of behavioral metrics and financial indicators.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Define target and selected features
target = 'Churn Value'
features = [
    'Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV',
    'ContractRisk', 'AutoPay', 'RiskExposure', 'ServiceCount'
]

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train model
logreg = LogisticRegression(max_iter=1000, penalty='l2', solver='lbfgs')
logreg.fit(X_train_scaled, y_train)

# Predict
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

# Evaluation Metrics
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

```

#### Logistic Regression Performance

**Classification Report:**

          precision    recall  f1-score   support

       0       0.82      0.90      0.86      1033
       1       0.64      0.47      0.54       374

accuracy                           0.79      1407

**Confusion Matrix (Actual vs Predicted):**

|            | Predicted 0 | Predicted 1 |
|------------|-------------|-------------|
| Actual 0   |    933      |    100      |
| Actual 1   |    198      |    176      |


### Output:
> Accuracy: 79%  
> AUC: 0.83  
> Key drivers: Contract Type, Monthly Charges, Risk Exposure

### ROC Curve – Logistic Regression

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='LogReg (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba)))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()
plt.show()
```

![ROC](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile5.png?raw=true) 

Interpretation:
The ROC curve shows strong performance, with an AUC of 0.83, indicating that the model is good at distinguishing churners from non-churners. The curve bows significantly toward the upper-left corner, which is a sign of effective classification.

---

### Step 5: XGBoost Comparison

1. Train XGBoost Model
   
```python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_proba_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

```
2. Model Performance

              precision    recall  f1-score   support

           0       0.82      0.89      0.85      1033
           1       0.61      0.46      0.52       374

    accuracy                           0.78      1407

**Confusion Matrix (Actual vs Predicted):**

|            | Predicted 0 | Predicted 1 |
|------------|-------------|-------------|
| Actual 0   |    922      |    111      |
| Actual 1   |    203      |    171      |
   
- AUC: 0.83  
- Feature importance dominated by `ContractRisk`

Interpretation:
- XGBoost achieved 78% accuracy, F1-score of 0.52 for churners, and ROC AUC of 0.83.
- It performs similarly to logistic regression in terms of overall AUC, but tends to slightly underperform in recall for the churn class compared to logistic regression.

3. ROC Curve – XGBoost

```python
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
plt.plot(fpr, tpr, label='XGBoost (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba_xgb)))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – XGBoost")
plt.legend()
plt.show()
```
![ROC](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile6.png?raw=true) 

Interpretation:
- The ROC curve again confirms a strong classifier. Both XGBoost and Logistic Regression deliver similar separation power between churners and retained customers.
- The model is appropriate for ranking churn risk, though additional tuning might improve recall.

4. Feature Importance Gain vs Weight – XGBoost

I compare feature importances using two different methods from the same trained XGBoost model:

📊 Plot 1: Feature Importance by gain
```python
xgb.plot_importance(xgb_model, importance_type='gain', title="XGBoost Feature Importance")
plt.show()
```
![importance by gain](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile7.png?raw=true) 

Interpretation:
- The 'gain' metric measures the average gain in accuracy a feature brings when it's used in a split.
- ContractRisk contributes 83% of the gain.
- This indicates ContractRisk has the most impact on model decisions.
- However, this view can be misleading if overused, as it may exaggerate the dominance of certain features.

📊 Plot 2: Feature Importance by weight
```python
import matplotlib.pyplot as plt
importances = xgb.get_booster().get_score(importance_type='weight')
importances = pd.DataFrame(importances.items(), columns=['Feature', 'F score'])
importances = importances.sort_values(by='F score', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importances['Feature'], importances['F score'])
plt.xlabel("F score")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()
```
![importance by weight](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile14.png?raw=true) 

Interpretation:
- The 'weight' metric measures the number of times a feature is used in a split across all trees.
- Provides a broader view of feature utility across the model.
- Monthly Charges, CLTV, and RiskExposure appear more balanced and relevant.
- This view is better for interpreting the model holistically and understanding breadth of use.

Reccomendation: For executive stakeholders, the weight version is more intuitive and supports actionable decisioning across features like CLTV, contract types, and service usage.

---

### ✅ SHAP Analysis

- Global SHAP: ContractRisk and RiskExposure increase churn
- Local SHAP: TotalCharges and AutoPay reduce churn

```python
!pip install shap --quiet
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# Rebuild TreeExplainer with trained XGBoost model
explainer = shap.Explainer(xgb_model, X_train)

# Compute SHAP values on test set
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values, max_display=10)
```
![shap](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile8.png?raw=true) 

Interpretation:
- Each dot represents a customer. Red = high feature value, blue = low feature value.
- SHAP value on the x-axis shows the impact on predicted churn probability.
- ContractRisk and RiskExposure are dominant drivers — high values increase churn risk.
- Monthly Charges also contributes significantly to churn.
- Tenure Months has a strong negative effect — longer-tenure customers are less likely to churn.

🔍 SHAP Force Plot – Individual Customer

```python
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])
```

![shap](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile9.png?raw=true) 

Interpretation:
- This explains a single prediction.
- The base value (here, around –1.658) represents the model’s average raw output (log-odds) across all customers.
- Features pushing to the left (blue) decrease the churn probability.
- Features pushing to the right (red) increase the churn probability.
- In this example:
- Total Charges = 4,542 (red): pushed the model toward a higher churn probability.
- ContractRisk = 0, RiskExposure = 1.266, ServiceCount = 5, and others (blue): pulled the prediction lower, toward retention.
- The final output (f(x) = –4.33) is below the base value, indicating the model predicted low churn probability for this customer.
  
---

### ✅ Score Binning + CLTV Simulation
In this step, I simulated a simple scoring strategy using the model’s predicted churn probabilities and customer CLTV values. This allows us to explore how churn risk and customer value interact to guide retention strategy.

🔢 Step 7.1: Binning Churn Scores
```python
df_score['Score'] = (1 - df_score['Churn_Prob']) * 600 + 300
df_score['Score'] = df_score['Score'].round()
```

📊 Step 7.2: Plot Score Bin vs Churn Risk & CLTV
```python
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Score Bin (High Risk → Low Risk)')
ax1.set_ylabel('Avg Churn Probability', color=color)
ax1.plot(score_summary['Score_Bin'], score_summary['Churn_Prob'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.invert_xaxis()

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Avg CLTV', color=color)
ax2.plot(score_summary['Score_Bin'], score_summary['CLTV'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Churn Score Bins vs CLTV")
plt.show()

```

![score bin](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile10.png?raw=true) 

🔍 Interpretation:
- Churn Risk decreases as the score bin increases — confirming model discrimination.
- CLTV increases with higher score bins — showing that high-value customers are also at higher risk.
- This insight is crucial for prioritizing retention strategies by targeting “high CLTV + high risk” customers first.

📉 Step 7.3: Estimate Total CLTV at Risk
I flagged the top 3 riskiest bins as high churn risk and estimated the total Customer Lifetime Value (CLTV) at risk if these customers churned.
```python
df_score['Retention_Flag'] = df_score['Score_Bin'].apply(lambda x: 1 if x <= 2 else 0)
total_risk_cltv = df_score[df_score['Retention_Flag'] == 1]['CLTV'].sum()
```
> ✅ $2,009,410 in total CLTV at risk (top 3 bins)

🧠 Insight:
- This simulation demonstrates how a model score can be operationalized into a simple retention strategy:
- Score Binning offers clear thresholds for prioritizing customer outreach.
- By combining model output with CLTV, the business can identify high-impact segments for retention interventions.

---

### ✅ Monitoring & Drift Detection (PSI Simulation)
In this step, I simulate slight distributional changes in key features to evaluate model stability and potential drift using PSI (Population Stability Index) analysis.

🔍 Objective:
Ensure the model remains reliable over time as customer behaviors shift.

🧬 Simulate Future Data
```python
# Copy test set
df_future = df_score.copy()

# Simulate slight distribution shift
np.random.seed(42)
df_future['Monthly Charges'] *= np.random.normal(1.02, 0.01, size=len(df_future))
df_future['Tenure Months'] *= np.random.normal(0.97, 0.02, size=len(df_future))
df_future['CLTV'] *= np.random.normal(0.99, 0.02, size=len(df_future))

# Generate churn scores from trained XGBoost model
df_future['Churn_Prob'] = xgb_model.predict_proba(df_future[X_test.columns])[:,1]
```

📊 Visualize Score & Feature Drift
```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Score distribution comparison
sns.kdeplot(df_score['Churn_Prob'], label="Current", ax=ax[0])
sns.kdeplot(df_future['Churn_Prob'], label="Future", ax=ax[0])
ax[0].set_title("Churn Score Distribution")
ax[0].legend()

# Monthly Charges drift example
sns.kdeplot(df_score['Monthly Charges'], label="Current", ax=ax[1])
sns.kdeplot(df_future['Monthly Charges'], label="Future", ax=ax[1])
ax[1].set_title("Monthly Charges Distribution")
ax[1].legend()

plt.tight_layout()
plt.show()
```
![drift](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile11.png?raw=true) 

Interpretation:
- The churn score distribution between current and simulated future data shows minimal drift, indicating good model stability.
- Slight shifts in Monthly Charges and Tenure Months were injected to mimic natural changes in usage or billing trends.
- Despite the shifts, the model maintained consistent predictions — verified using the PSI metric.

📏 PSI Example Output:
```python
def calculate_psi(expected, actual, bins=10):
    expected_percents, _ = np.histogram(expected, bins=bins, range=(0, 1), density=True)
    actual_percents, _ = np.histogram(actual, bins=bins, range=(0, 1), density=True)
    expected_percents += 1e-6  # avoid division by zero
    actual_percents += 1e-6
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi

psi_score = calculate_psi(df_score['Churn_Prob'], df_future['Churn_Prob'])
print(f"PSI: {psi_score:.4f}")
```

PSI: 0.0149

A PSI < 0.1 is considered stable.
This result suggests no material drift; model monitoring can proceed with confidence

---

### 🧩 Segmentation & Profiling
To support strategic decision-making, I created a churn-risk and value-based segmentation by classifying customers into four groups. This helps target retention efforts where they matter most.

🧮 Code: Risk-Value Segmentation
```python
# Churn threshold: top 30% as high risk
risk_threshold = df_score['Churn_Prob'].quantile(0.70)
value_threshold = df_score['CLTV'].median()

# Segment logic: combine churn risk with CLTV
def segment(row):
    if row['Churn_Prob'] >= risk_threshold:
        return 'High Churn - High Value' if row['CLTV'] >= value_threshold else 'High Churn - Low Value'
    else:
        return 'Low Churn - High Value' if row['CLTV'] >= value_threshold else 'Low Churn - Low Value'

df_score['segment'] = df_score.apply(segment, axis=1)

segment_summary = df_score.groupby('segment').agg({
    'CustomerID': 'count',
    'Churn_Prob': 'mean',
    'CLTV': 'mean'
}).rename(columns={'CustomerID': 'Count'}).reset_index()

segment_summary

```

| Segment                 | Count | Churn\_Prob | CLTV    |
| ----------------------- | ----- | ----------- | ------- |
| High Churn - High Value | 149   | 0.5746      | 5211.87 |
| High Churn - Low Value  | 273   | 0.6051      | 3210.51 |
| Low Churn - High Value  | 555   | 0.1191      | 5392.16 |
| Low Churn - Low Value   | 430   | 0.1425      | 3507.00 |

📌 Interpretation:
- High Churn - High Value (149 customers): These are the most strategically critical customers — high risk and high profitability. Retention campaigns should prioritize this group.
- High Churn - Low Value (273 customers): At-risk but lower value. Interventions should be cost-efficient (e.g., automated emails).
- Low Churn - High Value (555 customers): Loyal and valuable — ensure ongoing satisfaction to prevent future churn.
- Low Churn - Low Value (430 customers): Stable but lower value. No immediate action required.

📊 Customer Count by Segment (Bar Chart)
```python
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.barplot(data=segment_summary, x='Segment', y='Count')
plt.title('Customer Count by Segment')
plt.xticks(rotation=30)
plt.show()
```

![bar](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile12.png?raw=true) 

📌 Interpretation (Chart)
- This bar chart provides a clear view of how many customers fall into each of the four strategic segments:
- The largest segment is Low Churn – High Value, indicating a strong core customer base that is loyal and profitable.
- The second-largest is Low Churn – Low Value, a stable group that offers less revenue opportunity.
- Notably, High Churn – Low Value customers outnumber High Churn – High Value ones, which is helpful for prioritizing retention resources.
- High Churn – High Value is the smallest group in size but the most critical in impact — targeted efforts here could prevent the largest losses.

---

### 🔢 Step 10: Credit-Like Score Transformation

To make the model output more interpretable and aligned with traditional risk scoring systems, I transformed the churn probability into a credit-style score:
- Higher scores = lower risk (i.e., lower predicted churn probability)
- I scale from 300 to 900, resembling credit bureau formats

🧪 Code
```python
# Score transformation: invert churn prob (higher = lower risk)
df_score['score'] = (1 - df_score['Churn_Prob']) * 600 + 300
df_score['score'] = df_score['score'].round().astype(int)

plt.figure(figsize=(10,5))
sns.histplot(df_score['score'], bins=20, kde=False)
plt.title('Customer Score Distribution (300–900 Scale)')
plt.xlabel('Score')
plt.ylabel('Customer Count')
plt.show()
```
![scorecard](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/tmobile13.png?raw=true) 

📌 Interpretation:
- The plot shows the distribution of customer scores after transforming the churn probabilities.
- Most customers are concentrated on the high-score end (850–900), suggesting a majority are low-risk.
- A smaller but significant portion of customers falls below 600, indicating moderate to high churn risk.
- This transformation allows business teams to use familiar score ranges to segment customers, communicate risk to non-technical stakeholders, and set thresholds for retention interventions.

🧮 Score Binning & Risk Band Segmentation
To support targeted retention strategies and align model output with business action, I translated model scores into risk bands:
- Low Risk: score ≥ 750
- Moderate Risk: 600 ≤ score < 750
- High Risk: score < 600

This allows for strategic customer segmentation for resource allocation and communication.

🧪 Code
```python
# Assign risk bands from credit-style score
def assign_risk_band(score):
    if score >= 750:
        return 'Low Risk'
    elif score >= 600:
        return 'Moderate Risk'
    else:
        return 'High Risk'

df_score['Risk Band'] = df_score['score'].apply(assign_risk_band)

# Summarize churn probability and CLTV by risk group
risk_summary = df_score.groupby('Risk Band').agg({
    'CustomerID': 'count',
    'Churn_Prob': 'mean',
    'CLTV': 'mean'
}).rename(columns={'CustomerID': 'Count'}).reset_index()

risk_summary

```

| Risk Band     | Count | Churn\_Prob | CLTV    |
| ------------- | ----- | ----------- | ------- |
| High Risk     | 282   | 0.670394    | 3820.80 |
| Low Risk      | 786   | 0.081522    | 4648.59 |
| Moderate Risk | 339   | 0.368865    | 4195.99 |

📌 Interpretation:
- High Risk customers have a churn probability over 67% and relatively low CLTV, indicating urgent but cost-sensitive intervention.
- Low Risk customers have minimal churn probability and the highest CLTV, representing the most profitable and stable base.
- Moderate Risk customers occupy the middle ground, suggesting they could swing either way with targeted outreach.

---


## 🧠 Business Impact

This project demonstrates how churn risk modeling can support:

- Targeted retention
- Revenue risk assessment
- Strategic prioritization

---
