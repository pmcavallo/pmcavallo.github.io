---
layout: project
title: "NetworkIQ: Incident Risk Monitor (Render, Google Cloud, AWS)"
date: 2025-08-22
---

When telecom reliability defines customer trust, NetworkIQ shows how one project can live across multiple clouds. Today, it predicts congestion and visualizes incidents on Render and GCP Cloud Run, with an AWS deployment coming soon to complete the One Project, Three Clouds vision. Built with PySpark preprocessing, XGBoost prediction, and Streamlit dashboards, NetworkIQ demonstrates that portability, scalability, and explainability can be baked into a single AI-first system — no matter the platform.

**Live App:** [GCP Deployment](https://networkiq-49161466777.us-central1.run.app/) | [Render Deployment](https://network-iq.onrender.com/)  

---

## 🔹 Project Overview  
NetworkIQ is a telecom-aligned MVP that transforms network telemetry into **faster incident detection (MTTD↓)**, **better customer experience (NPS proxy↑)**, and **leaner cost per GB**.  
It demonstrates how **AI-first system design** can turn raw performance data into actionable insights for network operators.  

---

## 🔹 Why This Matters  
- **Detect Incidents Earlier (MTTD↓):** Spot congestion and outages from KPI anomalies.  
- **Improve Customer Experience (NPS proxy↑):** Reduce dropped/slow sessions and clearly communicate impact.  
- **Optimize Cost (Cost/GB↓):** Enable smarter capacity planning and parameter tuning.  

---

## 🔹 Tech Stack & Architecture  

**Architecture:**  

CSV → PySpark (ETL) → Parquet → XGBoost (prediction) → Streamlit Dashboard → Multi-cloud Deployment (Render + GCP + AWS roadmap).  

- **Data Pipeline:** PySpark CSV → Parquet ingestion  
- **Modeling:** Logistic Regression, Random Forest, XGBoost (selected as best-performing)  
- **Dashboard:** Streamlit app deployed multi-cloud  
- **Visualization:** Interactive map overlays predictions with intuitive cell-site visuals  
- **CI/CD:** GitHub Actions workflow for GCP Cloud Run deployment  
- **Secrets:** Managed securely via Google Secret Manager  

---

## 🔹 EDA & Key KPIs  

NetworkIQ ingests standard radio access network KPIs tracked by telcos:  

- **Throughput (Mbps):** Data delivery performance.  
- **Latency (ms):** Network responsiveness.  
- **Packet Loss (%):** Connection stability.  
- **Dropped Session Rate:** Customer experience proxy.

![NetworkIQ](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/iq1.png?raw=true)

**EDA Example (PySpark Snippet):**  

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("NetworkIQ").getOrCreate()

df = spark.read.csv("data/raw/sample_cells.csv", header=True, inferSchema=True)
df = df.withColumnRenamed("cell_id", "cell").withColumn("throughput_mbps", df["bytes"]/df["duration"])

df.show(5)
```  

---

## 🔹 Model Development  

### Model Performance  

| Model                | AUC   | KS   | Notes                                  |
|-----------------------|-------|------|----------------------------------------|
| Logistic Regression   | 0.74  | 0.28 | Interpretable but weaker baseline       |
| Random Forest         | 0.81  | 0.36 | Higher complexity, moderate gains       |
| **XGBoost**           | **0.86** | **0.42** | Best performance, robust & scalable    |  

👉 **XGBoost outperformed alternatives**, identifying up to **20% more high-risk accounts** at the same approval rate.  

**Training Example (Python):**  

```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
preds = xgb.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, preds)
print("AUC:", auc)
```  

![NetworkIQ](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/iq2.png?raw=true)

---

## 🔹 Deployment & Validation  

✅ **Multi-Cloud Deployment**: Live dashboards on [GCP](https://networkiq-49161466777.us-central1.run.app/) and [Render](https://network-iq.onrender.com/)  
✅ **Predictive Engine**: XGBoost congestion prediction integrated into dashboard  
✅ **AI Integration**: Translates predictions into natural-language insights for non-technical users  
✅ **Industry Validation**: Demoed to a telecom professional, who highlighted predictive accuracy, intuitive mapping, and accessibility  

**CI/CD Workflow (Excerpt):**  

```yaml
name: Deploy to Cloud Run

on:
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Authenticate
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}
    - name: Deploy
      run: gcloud run deploy networkiq --source . --region us-central1 --allow-unauthenticated
```  

---

## 🔹 Responsible AI & Monitoring  

- **Explainability:** SHAP values for feature contribution.  
- **Stability:** PSI to monitor population drift.  
- **Transparency:** Model card skeleton included in `/docs`.  
- **Ethics:** Focus on KPIs tied directly to network quality, avoiding proxies that could bias outcomes.  

---

## 🔹 Roadmap  

- **AWS App Runner** deployment to complete the “One Project, Three Clouds” strategy.  
- **Feedback Loops** for adaptive retraining.  
- **Advanced Forecasting** (ARIMA/Prophet baselines).  
- **Expanded KPI Coverage** for richer incident monitoring.  

---

## 🔹 Lessons Learned  

- **Spark Version Mismatch:** Fixed by aligning `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to Python 3.10 environment.  
- **API Latency:** Optimized Streamlit queries to reduce dashboard lag.  
- **Secrets Management:** Ensured API keys were never exposed in code or logs.  

---

## 🔹 License  
MIT © 2025 Paulo Cavallo.  

