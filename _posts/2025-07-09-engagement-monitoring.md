---
layout: post
title: 📶 Telecom Engagement Monitoring using Fractional Logistic Regression
---

This project implements a fractional logistic regression monitoring pipeline for tracking customer engagement in a telecom environment. It simulates realistic development and monitoring datasets to evaluate how well the model generalizes over time using key metrics such as RMSE, MAE, PSI, and calibration curves. The goal is to detect drift, miscalibration, or deterioration in model performance that may require retraining or recalibration, following a structured monitoring scorecard framework.

---

## 🎯 Objective

To monitor model stability and performance over time using:
- ✅ RMSE & MAE
- ✅ Calibration curves
- ✅ PSI (Population Stability Index)
- ✅ Vintage-level summaries

---

## 📁 Dataset Description

- `engagement_ratio` (target): fraction of days with customer activity in the month
- Simulated features include:
  - `age`, `tenure_months`, `avg_monthly_usage`, `network_issues`
  - Region and Plan Type (one-hot encoded)

Two datasets:
- **Development sample** (`2023Q4`)
- **Monitoring sample** (`2025Q2`)

---

## 🧪 Step 1: Load and Prepare Data

```python
import pandas as pd
import numpy as np

dev = pd.read_csv("engagement_dev_sample.csv")
mon = pd.read_csv("engagement_mon_sample.csv")

target = 'engagement_ratio'
features = [col for col in dev.columns if col != target]
```

---

## 🧮 Step 2: Fit Fractional Logistic Regression

We apply a **fractional logit transformation** to ensure predicted values remain in the (0, 1) interval:

    y* = [ y × (n - 1) + 0.5 ] / n

Where:

- `y` is the original target value (between 0 and 1),
- `n` is the number of observations,
- `y*` is the transformed target used in model fitting.



```python
import statsmodels.api as sm

def transform_fractional_y(y):
    n = len(y)
    return (y * (n - 1) + 0.5) / n

y_dev_frac = transform_fractional_y(dev[target])
X_dev_const = sm.add_constant(dev[features])

model = sm.GLM(y_dev_frac, X_dev_const, family=sm.families.Binomial(link=sm.families.links.Logit()))
result = model.fit()
dev["engagement_pred"] = result.predict(X_dev_const)
```

---

## 📈 Step 3: Predict on Monitoring Sample

```python
X_mon_const = sm.add_constant(mon[features])
y_mon_frac = transform_fractional_y(mon[target])
mon["engagement_pred"] = result.predict(X_mon_const)
```

---

## 📊 Step 4: RMSE and MAE

To evaluate the **accuracy** of the model’s predictions, I compute two key error metrics:

- **RMSE (Root Mean Squared Error)** measures the square root of the average squared differences between predicted and actual values. It is more sensitive to large errors.
- **MAE (Mean Absolute Error)** measures the average of absolute differences between predictions and actual outcomes. It treats all errors equally, regardless of magnitude.

These metrics give a quantitative view of the model's predictive precision in both development and monitoring samples.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_dev = mean_squared_error(dev[target], dev["engagement_pred"])
rmse_dev = np.sqrt(mse_dev)
mse_mon = mean_squared_error(mon[target], mon["engagement_pred"])
rmse_mon = np.sqrt(mse_mon)

mae_dev = mean_absolute_error(dev[target], dev["engagement_pred"])
mae_mon = mean_absolute_error(mon[target], mon["engagement_pred"])
```

---

## 📉 Step 5: PSI (Population Stability Index)

The **Population Stability Index (PSI)** is used to measure how much the distribution of model scores or inputs has shifted over time between the development and monitoring datasets.

- It helps detect **data drift** or **changes in customer behavior** that could impact model performance.
- A PSI below **0.10** generally indicates stability.
- Values between **0.10 and 0.25** suggest moderate shift.
- Values above **0.25** signal a significant change and may warrant investigation or model redevelopment.

This step ensures that the population on which the model is applied remains representative of the original training sample.

#### 📐 PSI Formula

For each bin *i*, compare the share of records in development vs. monitoring:

PSI = Σ (pᵢ − qᵢ) × ln(pᵢ / qᵢ)

Where:
- pᵢ = proportion in bin *i* from the **development** sample
- qᵢ = proportion in bin *i* from the **monitoring** sample


```python
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    psi_values = (expected_counts - actual_counts) * np.log((expected_counts + 1e-5) / (actual_counts + 1e-5))
    return np.sum(psi_values)

psi_score = calculate_psi(dev["engagement_pred"], mon["engagement_pred"])

buckets = np.linspace(0, 1, 11)
dev_dist = np.histogram(dev['engagement_pred'], bins=buckets)[0] / len(dev)
mon_dist = np.histogram(mon['engagement_pred'], bins=buckets)[0] / len(mon)

fig2, ax2 = plt.subplots()
bar_width = 0.4
ax2.bar(buckets[:-1], dev_dist, width=bar_width, label='Development', align='edge')
ax2.bar(buckets[:-1] + bar_width, mon_dist, width=bar_width, label='Monitoring', align='edge')
ax2.set_title("PSI Bucket Comparison")
ax2.set_xlabel("Score Buckets")
ax2.set_ylabel("Proportion")
ax2.legend()
fig2.savefig("psi_distribution.png", bbox_inches='tight')

```
![PSI Distribution](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/psi_distribution.png?raw=true) 

### 📘 Interpretation: PSI Chart

- The distribution of predicted scores in both samples is fairly consistent.
- Minor deviations exist in some buckets, but the PSI value (e.g., ~0.087) remains below the typical threshold of 0.1.
- ✅ No significant input drift detected. Model score distributions remain stable.

### 📊 PSI by Variable

We evaluate **population stability** for key model inputs and outputs between the development (2023Q4) and monitoring (2025Q2) datasets.

This helps detect **data drift** or **segment shift** that may impact model validity.

---

#### 📋 PSI Table

| Variable            |   PSI   | Status          |
|---------------------|--------:|------------------|
| engagement_pred     |  0.0807 | ✅ Stable        |
| engagement_ratio    |  0.0544 | ✅ Stable        |
| age                 |  0.0313 | ✅ Stable        |
| tenure_months       |  0.1493 | ⚠️ Moderate Shift |
| avg_monthly_usage   |  0.1147 | ⚠️ Moderate Shift |
| network_issues      |  0.0154 | ✅ Stable        |

---

#### 🧠 Interpretation

- ✅ **Stable (< 0.10)**: No significant shift in distribution  
- ⚠️ **Moderate Shift (0.10–0.25)**: Monitor carefully  
- 🛑 **Major Shift (> 0.25)**: Model likely impacted — investigate further



---

#### 🧪 Code Used to Generate PSI Table

```python
variables = [
    'engagement_pred', 'engagement_ratio', 'age',
    'tenure_months', 'avg_monthly_usage', 'network_issues'
]

psi_results = []

for var in variables:
    if var in dev.columns and var in mon.columns:
        psi_value = calculate_psi(dev[var], mon[var])
        
        if psi_value < 0.10:
            status = "✅ Stable"
        elif psi_value < 0.25:
            status = "⚠️ Moderate Shift"
        else:
            status = "🔁 Major Shift"
        
        psi_results.append({
            "Variable": var,
            "PSI": round(psi_value, 4),
            "Status": status
        })
    else:
        print(f"⚠️ Skipping variable '{var}' — not found in both datasets.")
```
---

## 🧮 Step 6: Calibration Curves

Calibration curves compare **predicted engagement probabilities** to **actual observed outcomes** across deciles.

This helps assess how well the model's probability estimates reflect reality.  
- A well-calibrated model will have points close to the **45° diagonal** line.
- Separate curves for **development** and **monitoring** datasets highlight whether calibration has drifted over time.

Maintaining good calibration ensures model predictions remain interpretable and trustworthy for decision-making.

```python
dev["quantile"] = pd.qcut(dev["engagement_pred"], 10, labels=False)
mon["quantile"] = pd.qcut(mon["engagement_pred"], 10, labels=False)

calib_dev = dev.groupby("quantile")[[target, "engagement_pred"]].mean()
calib_mon = mon.groupby("quantile")[[target, "engagement_pred"]].mean()

import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
ax1.plot(calib_dev['engagement_pred'], calib_dev['engagement_ratio'], marker='o', label='Development')
ax1.plot(calib_mon['engagement_pred'], calib_mon['engagement_ratio'], marker='x', label='Monitoring')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6)
ax1.set_title("Calibration Curve by Decile")
ax1.set_xlabel("Average Predicted Engagement")
ax1.set_ylabel("Average Actual Engagement")
ax1.legend()
fig1.savefig("calibration_curve.png", bbox_inches='tight')
```

![Calibration Curve](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/calibration_curve.png?raw=true) 

### 📘 Interpretation: Calibration Curve

- The development and monitoring curves both closely track the 45° reference line.
- This indicates that predicted engagement probabilities are well-aligned with observed values.
- ✅ The model remains well-calibrated and consistent across both time periods.

---

## 🗓️ Step 7: Vintage-Level Comparison

```python
dev["vintage"] = "2023Q4"
mon["vintage"] = "2025Q2"
vintages = pd.concat([dev, mon])
vintage_table = vintages.groupby("vintage")[[target, "engagement_pred"]].mean()
```
### 🗓️ Step 7: Vintage-Level Comparison

The table below compares average **actual** and **predicted** engagement across development and monitoring vintages:

| **Vintage** | **Average Actual Engagement** (`engagement_ratio`) | **Average Predicted Engagement** (`engagement_pred`) |
|-------------|-----------------------------------------------------|--------------------------------------------------------|
| 2023Q4      | 0.7034                                              | 0.7034                                                 |
| 2025Q2      | 0.7256                                              | 0.7267                                                 |

---

#### ✅ Interpretation:

- The model remains well-calibrated across vintages — the predicted engagement is extremely close to the actual engagement in both time periods.
- Notably, engagement increased in **2025Q2**, and the model accurately captured this upward shift.
- The alignment between actual and predicted averages suggests strong **temporal stability** in the model's behavior, reducing the need for immediate recalibration.

> 📌 Conclusion: Vintage-level performance is consistent, and the model appears robust across time.

---


## 🧾 Results Summary

```python
summary = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "PSI"],
    "Development": [rmse_dev, mae_dev, None],
    "Monitoring": [rmse_mon, mae_mon, psi_score]
})
summary["Development"] = summary["Development"].apply(lambda x: round(x, 4) if pd.notnull(x) else "")
summary["Monitoring"] = summary["Monitoring"].apply(lambda x: round(x, 4) if pd.notnull(x) else "")
print(summary)
```


| Metric     | Development | Monitoring |
|------------|-------------|------------|
| **RMSE**   | 0.0504      | 0.0501     |
| **MAE**    | 0.0401      | 0.0399     |
| **PSI**    | 0.0874      |            |

#### ✅ Interpretation:

- **RMSE** and **MAE** are virtually unchanged between development and monitoring. This indicates that the model's accuracy and error distribution have remained stable over time. There is no evidence of deteriorating prediction quality.

- **PSI** is calculated at **0.0874**, which is **below the 0.10 stability threshold** commonly used in production models. This suggests that the distribution of predicted engagement scores has not shifted significantly between samples, and the underlying customer population remains stable.

---

#### 🟢 Conclusion:

All three indicators suggest the model continues to perform reliably and is **well-calibrated** across both timeframes. **There is no immediate need for redevelopment or recalibration** based on this monitoring cycle. But below I'll produce a scorecard and a final score to drive reccomendation. 

---

## ⚠️ Why I Don’t Use MAPE

**MAPE (Mean Absolute Percentage Error)** is not used for fractional targets because:

- ⚠️ `engagement_ratio` can be 0
- ⚠️ Division by zero leads to instability
- ✅ RMSE and MAE offer robust, interpretable alternatives

---

## 📋 Final Monitoring Scorecard


```python
def compute_monitoring_score(rmse_dev, rmse_mon, mae_dev, mae_mon, psi_score,
                             rmse_thresh=0.02, mae_thresh=0.02, psi_thresh=0.1,
                             w_rmse=0.4, w_mae=0.4, w_psi=0.2):
    # Normalize deltas
    rmse_delta = max(0, (rmse_mon - rmse_dev) / rmse_thresh)
    mae_delta = max(0, (mae_mon - mae_dev) / mae_thresh)
    psi_norm = max(0, psi_score / psi_thresh)

    # Clip values at 1 to bound score
    rmse_score = min(rmse_delta, 1)
    mae_score = min(mae_delta, 1)
    psi_score_adj = min(psi_norm, 1)

    # Weighted average score
    final_score, recommendation = compute_monitoring_score(
    rmse_dev=rmse_dev,
    rmse_mon=rmse_mon,
    mae_dev=mae_dev,
    mae_mon=mae_mon,
    psi_score=psi_score
)
```

### 🧮 Final Monitoring Score: How It Works

To summarize the model’s health into a single, interpretable metric, we calculate a **Final Monitoring Score** using a weighted combination of key evaluation metrics:

- **RMSE Drift** (change in prediction error)
- **MAE Drift**
- **PSI Score** (population stability)

Each component is scaled between 0 and 1 and multiplied by a weight reflecting its importance. For example:

```python
score = (
    0.4 * normalized_rmse_drift +
    0.4 * normalized_mae_drift +
    0.2 * psi_score
)
```

| Metric      | Score     |
|-------------|-----------|
| RMSE Delta  | ~0.00     |
| MAE Delta   | ~0.00     |
| PSI Drift   | 0.87      |

**📊 Final Score: `0.175`**

### 👉 Recommendation:
**✅ Model is stable — no action needed**

### 🟢 What the Final Monitoring Score Means

| **Final Score Range** | **Interpretation**                | **Recommended Action**     |
|------------------------|----------------------------------|-----------------------------|
| **0.00 – 0.30**        | Excellent stability              | ✅ No action needed         |
| **0.31 – 0.60**        | Moderate drift                   | ⚠️ Monitor closely          |
| **0.61 – 1.00**        | Significant performance change   | 🔁 Consider redevelopment   |

### 🧠 Rationale Behind the Final Score Thresholds

The final monitoring score aggregates multiple sources of model degradation (e.g., error drift and population shift) into a single value between 0 and 1. The thresholds for interpreting this score are based on common practices in model monitoring, sensitivity to drift, and real-world operational risk.

---

#### ✅ Threshold: `0.00 – 0.30` → **Excellent Stability**
- This range reflects minimal changes in both error metrics (RMSE/MAE) and score distribution (PSI).
- A score in this range indicates the model is performing similarly to how it did during development.
- No retraining or recalibration is required; continue periodic monitoring as planned.

---

#### ⚠️ Threshold: `0.31 – 0.60` → **Moderate Drift**
- This zone represents small but noticeable deviations from development benchmarks.
- While model accuracy may still be acceptable, increasing drift could signal early warning signs of deterioration.
- Action: Monitor more frequently and investigate potential drivers of change (e.g., behavior, policy, or data input changes).

---

#### 🔁 Threshold: `0.61 – 1.00` → **Significant Performance Change**
- A score in this range suggests that model predictions are diverging meaningfully from actual outcomes, or that the input data has shifted materially.
- Such a shift poses risk to business decisions relying on the model’s output.
- Action: Initiate model review, backtesting, and consider redevelopment or recalibration.

---

> These thresholds serve as **guidelines**, not hard rules. They balance statistical rigor with operational practicality and are flexible enough to adapt across use cases.

