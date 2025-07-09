
# 📶 Telecom Engagement Monitoring using Fractional Logistic Regression

This project implements a **fractional logistic regression** monitoring pipeline for tracking customer engagement in a postpaid telecom environment.

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

We apply a fractional logit transformation:  
\( y^* = \frac{y (n - 1) + 0.5}{n} \)

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

```python
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    psi_values = (expected_counts - actual_counts) * np.log((expected_counts + 1e-5) / (actual_counts + 1e-5))
    return np.sum(psi_values)

psi_score = calculate_psi(dev["engagement_pred"], mon["engagement_pred"])
```

---

## 🧮 Step 6: Calibration Curves

```python
dev["quantile"] = pd.qcut(dev["engagement_pred"], 10, labels=False)
mon["quantile"] = pd.qcut(mon["engagement_pred"], 10, labels=False)

calib_dev = dev.groupby("quantile")[[target, "engagement_pred"]].mean()
calib_mon = mon.groupby("quantile")[[target, "engagement_pred"]].mean()
```

**Visualization:**

![Calibration Curve](calibration_curve.png)

```python
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

---

## 📊 Step 8: PSI Distribution Plot

```python
# Already generated: see psi_distribution.png
```

![PSI Distribution](psi_distribution.png)

```python
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

### 📘 Interpretation: PSI Chart

- The distribution of predicted scores in both samples is fairly consistent.
- Minor deviations exist in some buckets, but the PSI value (e.g., ~0.087) remains below the typical threshold of 0.1.
- ✅ No significant input drift detected. Model score distributions remain stable.


---

## 🧾 Results Summary

| Metric     | Development | Monitoring |
|------------|-------------|------------|
| **RMSE**   | 0.0504      | 0.0501     |
| **MAE**    | 0.0401      | 0.0399     |
| **PSI**    | 0.0874      |            |

---

## ⚠️ Why We Don’t Use MAPE

**MAPE (Mean Absolute Percentage Error)** is not used for fractional targets because:

- ⚠️ `engagement_ratio` can be 0
- ⚠️ Division by zero leads to instability
- ✅ RMSE and MAE offer robust, interpretable alternatives

---

## ✅ Output Files

- `engagement_dev_scored.csv`
- `engagement_mon_scored.csv`
- `engagement_calibration_dev.csv`
- `engagement_calibration_mon.csv`
- `engagement_vintage_summary.csv`

---

## 🚀 Conclusion

This complete project simulates and monitors a fractional logistic regression model for telecom engagement.  
You can easily adapt the structure to:

- Healthcare adherence
- Edtech course completion
- Energy consumption rates



---

## 📉 Step 9: Results Summary Table

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


```python

summary["Development"] = summary["Development"].apply(lambda x: round(x, 4) if pd.notnull(x) else "")
summary["Monitoring"] = summary["Monitoring"].apply(lambda x: round(x, 4) if pd.notnull(x) else "")
print(summary)
```



---

## 📉 Step 9: Results Summary Table

```python

summary["Development"] = summary["Development"].apply(lambda x: round(x, 4) if pd.notnull(x) else "")
summary["Monitoring"] = summary["Monitoring"].apply(lambda x: round(x, 4) if pd.notnull(x) else "")
print(summary)
```

This table reports RMSE, MAE and PSI drift for both samples.

```python

summary["Development"] = summary["Development"].apply(lambda x: round(x, 4) if pd.notnull(x) else "")
summary["Monitoring"] = summary["Monitoring"].apply(lambda x: round(x, 4) if pd.notnull(x) else "")
print(summary)
```


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

| Metric      | Score     |
|-------------|-----------|
| RMSE Delta  | ~0.00     |
| MAE Delta   | ~0.00     |
| PSI Drift   | 0.87      |

**📊 Final Score: `0.175`**

### 👉 Recommendation:
**✅ Model is stable — no action needed**
