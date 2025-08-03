---
layout: post
title: 📘 AI-Augmented BNPL Risk Dashboard with Intelligent Override System
--- 


This project builds a real-time BNPL risk monitoring dashboard with intelligent override logic, powered by anomaly detection, adaptive policy simulation, and auto-deployment via Render. The dashboard mimics intelligent, data-responsive policy decisions. It serves as a template for modern credit risk monitoring pipelines with explainable AI and modular automation.

---

## 🧠 Intelligent Component: The Soul of the Project

### Adaptive Policy Override Logic
- Detects anomalous default behavior in low-risk segments.
- Triggers an override policy when anomalies surpass a threshold.
- Simulates dynamic segment reclassification to high risk.
- Automatically integrates with Streamlit for explainability and action.

```python
# Sample Trigger Logic
anomaly_count = pivot_table["low_risk_anomaly"].sum()
threshold = 3
policy_flag = anomaly_count > threshold
```

```python
# Override Simulation
anomaly_bins = pivot_table[pivot_table["low_risk_anomaly"] == True]["score_bin"].tolist()
override_df = pd.DataFrame({
    "score_bin": anomaly_bins,
    "override_high_risk": True,
    "reason": "Low-risk default rate exceeded high-risk segment"
})
```

---

## 🧼 Data Preprocessing Summary

- Simulated missing values: income, late_payment_count, risk_segment.
- Applied MCAR assumption for simplicity.
- Imputation methods: mean, median, and mode respectively.
- Winsorization at 1st and 99th percentiles to handle outliers.
- Created clean and winsorized datasets for modeling.

---

## 🧪 Feature Engineering

- Target: `flag_ever_90plus` derived from delinquency logic.
- Key features: `late_payment_count`, `late_payment_ratio`, `has_prior_delinquency`.
- VIF analysis used to assess multicollinearity — all retained for model richness.

---

## 🧠 Expanded Intelligent Override Steps and Interpretations

```python
# --- Imports ---
import pandas as pd
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# ✅ Step 0: Sanity Check using the correct column names from df_missing
required_columns = {'score_bin', 'risk_segment', 'defaulted'}

if 'df_missing' not in globals():
    raise ValueError("❌ DataFrame `df_missing` is not defined in the environment.")

missing_columns = required_columns - set(df_missing.columns)
if missing_columns:
    raise ValueError(f"❌ `df_missing` is missing required columns: {missing_columns}")

if df_missing.empty:
    raise ValueError("❌ `df_missing` is empty.")

print("✅ Sanity check passed: df_missing contains all required columns and is not empty.")

# ✅ Step 1: Group and summarize
segment_score_summary = (
    df_missing
    .groupby(['score_bin', 'risk_segment'])
    .agg(default_rate=('defaulted', 'mean'))
    .reset_index()
)

# ✅ Step 2: Create pivot table
pivot_table = (
    segment_score_summary
    .pivot_table(index='score_bin', columns='risk_segment', values='default_rate')
    .round(3)
    .fillna("N/A")
)

# ✅ Step 3: Construct prompt
prompt_text = (
    "Summarize the following table of default rates by score bin and risk segment. "
    "Highlight any unusual findings:\n\n"
    + pivot_table.to_string()
)

# ✅ Step 4: Deploy LLM
model_id = "huggingface-text2text-flan-t5-large"
role = sagemaker.get_execution_role()

llm = JumpStartModel(
    model_id=model_id,
    role=role
)

predictor = llm.deploy(
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    initial_instance_count=1,
    instance_type="ml.m5.large"  # or ml.t2.medium if needed
)

# ✅ Step 5: Generate LLM Summary
response = predictor.predict({"inputs": prompt_text})

print("\n📄 LLM-Generated Summary:\n")
print(response[0]['generated_text'])

# ✅ Cleanup
predictor.delete_endpoint()
```

![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima.png?raw=true) 


```python
# ✅ Sanity Check: Ensure required columns are present
required_columns = {'score_bin', 'default_rate'}
missing_columns = required_columns - set(segment_score_summary.columns)

if missing_columns:
    raise ValueError(f"❌ `segment_score_summary` is missing required columns: {missing_columns}")
if segment_score_summary.empty:
    raise ValueError("❌ `segment_score_summary` is empty.")
print("✅ Sanity check passed: segment_score_summary contains all required columns and is not empty.")

# 🧮 Pivot Table (no segmentation)
pivot_table = segment_score_summary.set_index("score_bin")
display(pivot_table)

# 🧠 Simulated LLM Summary Generation
summary_prompt = """
Summarize the following table of default rates by score bin.
Highlight any unusual findings or patterns in the default rate as the score_bin increases.
"""

print("------------\n🧠 LLM-Generated Summary:\n")
print(summary_prompt)
print(pivot_table)

# ✅ Output Verification (basic hallucination guard)
expected_bins = set(segment_score_summary['score_bin'].unique())
expected_values = set(segment_score_summary['default_rate'].round(3))

hallucinated = False
output_text = "Identifying and summarizing default trends by score bin."

# Regex check (for float values that might not match)
import re
found_values = set(map(float, re.findall(r"0\.\d{3}", output_text)))
if not found_values.issubset(expected_values):
    hallucinated = True

if hallucinated:
    print("⚠️ Potential hallucination(s) detected in LLM summary:")
    print("Values not found in source table:", found_values - expected_values)
else:
    print("✅ LLM output validated: All numerical references match the input table.")
```

![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima.png?raw=true) 

```python
# ✅ Intelligent Check: Flag score bins where low-risk default rate exceeds high-risk
import numpy as np
import pandas as pd

# Pivot the table: rows = score_bin, columns = risk_segment, values = default_rate
pivot_table = segment_score_summary.pivot_table(
    index='score_bin',
    columns='risk_segment',
    values='default_rate'
).reset_index()

# Optional: Fill NaNs with 0 just in case (in real life, be cautious with this)
pivot_table = pivot_table.fillna(0)

# Add flag: where low-risk default rate > high-risk default rate
pivot_table["low_risk_anomaly"] = pivot_table["low"] > pivot_table["high"]

# ✅ Output summary: show only flagged rows
anomalies_detected = pivot_table[pivot_table["low_risk_anomaly"] == True]

if not anomalies_detected.empty:
    print("⚠️ Detected potential anomalies: Low-risk segments defaulting more than high-risk in the following score bins:")
    display(anomalies_detected[["score_bin", "low", "high", "low_risk_anomaly"]])
else:
    print("✅ No low-risk anomalies detected. Segment behavior is consistent with expected risk ordering.")

# Optionally merge the flag back into segment_score_summary for full visibility
segment_score_summary = segment_score_summary.merge(
    pivot_table[["score_bin", "low_risk_anomaly"]],
    on="score_bin",
    how="left"
)
```

![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima.png?raw=true) 

```python
# ✅ Intelligent trigger: if anomalies exceed threshold, take action
anomaly_count = pivot_table["low_risk_anomaly"].sum()
threshold = 3

print(f"\n🧠 {anomaly_count} anomalies detected (low-risk > high-risk).")

if anomaly_count > threshold:
    policy_flag = True
    print("⚡ Action: Adaptive policy triggered. Consider retraining, segment override, or risk policy adjustment.")
else:
    policy_flag = False
    print("✅ No policy action required. Anomaly level is within tolerance.")

# Optionally add to master monitoring summary
segment_score_summary["policy_trigger"] = policy_flag
```
![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima.png?raw=true) 

```python
# ✅ Identify score_bins where low-risk default rate > high-risk
anomaly_bins = pivot_table[pivot_table["low_risk_anomaly"] == True]["score_bin"].tolist()

# ✅ Simulate policy override: elevate segment to high risk in those bins
override_df = pd.DataFrame({
    "score_bin": anomaly_bins,
    "override_high_risk": True,
    "reason": "Low-risk default rate exceeded high-risk segment"
})

# ✅ Display the override simulation table
override_df
```
![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima.png?raw=true) 

```python
# Merge override simulation with original default rates
impact_df = override_df.merge(segment_score_summary, on="score_bin", how="left")

# Estimate expected defaults under baseline and override scenarios
impact_df["baseline_default"] = impact_df["default_rate"]  # original (low-risk) default rate
impact_df["override_default"] = segment_score_summary[
    segment_score_summary["risk_segment"] == "high"
].set_index("score_bin").loc[impact_df["score_bin"], "default_rate"].values

# Calculate impact
impact_df["net_change"] = impact_df["override_default"] - impact_df["baseline_default"]

# Select and order relevant columns
override_policy_impact_summary = impact_df[[
    "score_bin",
    "baseline_default",
    "override_default",
    "net_change",
    "reason"
]]

# Round for display
override_policy_impact_summary = override_policy_impact_summary.round(6)

# Display
override_policy_impact_summary
```


```python
from datetime import datetime
from IPython.display import Markdown, display

# Compose the strategy brief
date_str = datetime.today().strftime("%B %d, %Y")
policy_brief = f"""
# 📋 Strategy Policy Brief – Adaptive Risk Policy Trigger
**Date:** {date_str}

## 🧠 Summary
Anomalies were detected in default behavior:
Low-risk segments showed higher default rates than high-risk segments in **8 score bins**.

## ⚠️ Triggered Action
- **Policy Flag Activated:** `True`
- **Trigger Logic:** `low-risk default rate > high-risk` in ≥ 3 bins
- **Impacted Bins:** {", ".join(str(bin) for bin in override_df['score_bin'].tolist())}

## 🔁 Simulated Override
The following score bins had their **risk segment reclassified to High-Risk**:

{override_df.to_markdown(index=False)}

## 📌 Rationale
> "Low-risk default rate exceeded high-risk segment."
This override supports proactive protection of the portfolio by adapting to behavioral drift.

## 🧭 Recommendations
- ✅ Continue tracking drift patterns quarterly
- 🔄 Retrain score model if anomaly pattern persists
- 🧪 Test alternate segmentation criteria if overrides repeat
- 🗂️ Store override table as part of policy traceability
```

# 🧾 Strategy Policy Brief – Adaptive Risk Policy Trigger  
**Date:** August 02, 2025  

---

## 🧠 Summary  
Anomalies were detected in default behavior:  
Low-risk segments showed higher default rates than high-risk segments in **8 score bins**.

---

## ⚡ Triggered Action  

- **Policy Flag Activated:** `True`  
- **Trigger Logic:** `low-risk default rate > high-risk` in ≥ 3 bins  
- **Impacted Bins:** 0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0  

---

## 🛠️ Simulated Override  

The following score bins had their **risk segment reclassified to High-Risk**:

| score_bin | override_high_risk | reason                                              |
|-----------|--------------------|------------------------------------------------------|
| 0         | True               | Low-risk default rate exceeded high-risk segment     |
| 1         | True               | Low-risk default rate exceeded high-risk segment     |
| 2         | True               | Low-risk default rate exceeded high-risk segment     |
| 3         | True               | Low-risk default rate exceeded high-risk segment     |
| 4         | True               | Low-risk default rate exceeded high-risk segment     |
| 7         | True               | Low-risk default rate exceeded high-risk segment     |
| 8         | True               | Low-risk default rate exceeded high-risk segment     |
| 9         | True               | Low-risk default rate exceeded high-risk segment     |

---

## 🧩 Rationale  

> “Low-risk default rate exceeded high-risk segment.”  
This override supports proactive protection of the portfolio by adapting to behavioral drift.

---

## ✅ Recommendations  

- 🟢 **Continue tracking drift patterns quarterly**  
- 🔁 **Retrain score model if anomaly pattern persists**  
- 🧪 **Test alternate segmentation criteria if overrides repeat**  
- 📦 **Store override table as part of policy traceability**

---

_This report was generated by the intelligent policy system to support risk governance decisions._



---

## 📊 Dashboard and Visuals (Streamlit)

File: `streamlit_dashboard.py` (deployed via Render)

- Upload interface for `segment_score_summary.csv` and `override_df.csv`
- Key sections:
  1. Default Rate by Score Bin
  2. Policy Trigger
  3. Anomaly Table
  4. Override Table
  5. Score Trends Over Time
  6. Override Volumes
  7. Approval Rates
  8. Risk Alerts

```python
# Sample line plot code
sns.lineplot(data=segment_df, x="score_bin", y="default_rate", hue="risk_segment")
```

---

## 🚀 Deployment

- GitHub repo updated with new visuals and override logic.
- Render deployment: [Live App](https://bnpl-risk-dashboard.onrender.com/)
- Auto-redeploys on each push to `main`.

---

## 🧪 Local Testing Instructions

1. Clone the GitHub repo:
```bash
git clone https://github.com/your_username/BNPL-Risk-Dashboard.git
cd BNPL-Risk-Dashboard
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the dashboard:
```bash
streamlit run streamlit_dashboard.py
```

5. Upload the sample files:
   - `segment_score_summary.csv`
   - `override_df.csv`

---

## 🧾 Included Files

- `streamlit_dashboard.py` – Full app logic with intelligent override visuals
- `requirements.txt` – All required packages for deployment
- `render.yaml` – Render app configuration
- `segment_score_summary.csv` – Sample input for segment analysis
- `override_df.csv` – Sample override simulation output

---

## 📌 Final Thoughts

This dashboard is more than a visualization tool.

It mimics intelligent, data-responsive policy decisions. It serves as a template for modern credit risk monitoring pipelines with explainable AI and modular automation.





# Display in notebook
display(Markdown(policy_brief))
```
