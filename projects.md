---
layout: default
title: Projects
permalink: /projects/
---

A selection of hands-on projects demonstrating real-world data science, modeling, and cloud deployment; built with Python, scikit-learn, **PySpark**, **XGBoost/CatBoost**, **SHAP**, and shipped via **Streamlit/Render** and **AWS** (S3, SageMaker, Lambda, MWAA/Airflow), with visuals in **Tableau**.

---
**BNPL Credit Risk Insights Dashboard (Python + Streamlit)**

A hands-on, end-to-end BNPL risk project that turns raw lending/repayment data into an interactive decision dashboard. It demonstrates modern risk workflows—from feature engineering and modeling to monitoring and “what-if” policy simulation—built for clarity, speed, and explainability.

**Highlights**
- **Data & Features:** Synthetic BNPL portfolio with engineered signals (loan-to-income, usage rate, delinquency streaks, tenure, interactions).
- **Modeling & Explainability:** Regularized logistic/CatBoost scoring with calibration, AUC/KS, and SHAP to validate driver logic.
- **Monitoring:** Drift/stability checks (e.g., PSI), score distribution tracking, and cohort comparisons across risk segments.
- **Policy Sandbox:** Threshold sliders to simulate approval/charge-off trade-offs, segment impacts, and recalibration scenarios.

📌 **Business Impact:** Helps risk teams test policies before rollout, quantify approval vs. losses, and document governance-ready decisions.

🔗 [View Full Project](https://pmcavallo.github.io/BNPL-Risk-Dashboard/)

---

**Credit Risk Model Deployment & Monitoring (AWS + PySpark + CatBoost)**

This flagship project showcases an end-to-end credit risk modeling pipeline — from scalable data processing to cloud deployment — aligned with best practices in financial services. Built using PySpark, CatBoost, SHAP, and AWS (S3, CLI), it simulates how modern risk pipelines are deployed and monitored at scale.

The full solution includes:

- **PySpark ETL pipeline** to preprocess large-scale synthetic telecom-style credit data, with engineered risk features (CLTV, utilization bins, delinquency flags)
- **Distributed logistic regression** using PySpark MLlib to validate scalable modeling workflows and evaluate performance using AUC and KS
- **AWS S3 integration** to export Parquet-formatted model-ready data for cloud-based storage and future MLOps orchestration
- **CatBoost modeling** to improve predictive power with categorical support and built-in regularization
- **SHAP explainability** to verify that key drivers (e.g., FICO, CLTV) align with domain logic and are not proxies or artifacts
- **Segment-level analysis** comparing predicted vs actual default rates by state, identifying under- and over-prediction patterns
- **Business recommendations** on scorecard calibration, behavioral feature expansion, and future automation (e.g., Airflow, SageMaker Pipelines)

💼 **Business Impact**: This project simulates a realistic production-grade credit risk pipeline — bridging data engineering, ML modeling, and cloud deployment. It highlights how interpretability and geographic segmentation can inform policy, governance, and model recalibration.

📁 [View Full Project](https://pmcavallo.github.io/aws-flagship-project/)

---

### Telecom Churn Modeling & Retention Strategy

This project demonstrates how predictive modeling and customer segmentation can be used to drive retention strategy in a telecom context. Using a publicly available customer dataset, I developed a full churn risk pipeline.

The final solution integrates:

- **Churn prediction modeling** using Logistic Regression and XGBoost with performance comparisons (AUC ≈ 0.83)
- **SHAP explainability** to identify key churn drivers (e.g., Contract Type, Risk Exposure)
- **Scorecard simulation** converting churn risk into a 300–900 scale for business-friendly deployment
- **Customer lifetime value (CLTV) integration** to quantify revenue risk across risk bands
- **Segmentation framework** (High Churn–High Value, Low Churn–Low Value, etc.) for targeted retention campaigns
- **Drift monitoring** using Population Stability Index (PSI) to track score performance over time

💡 **Business Impact**: The project enables strategic prioritization by identifying high-risk, high-value customers at risk of churn, supporting proactive retention efforts, revenue protection, and long-term profitability.

👉 [View Full Project](https://pmcavallo.github.io/Churn-Modeling-Complete)


---

### Telecom Customer Segmentation with Python

**Objective:**  
Developed a customer segmentation model using unsupervised learning on simulated postpaid telecom data to identify actionable behavioral clusters for marketing, retention, and product strategy.

**Highlights:**
- Simulated 5,000 realistic customer profiles with usage, support, contract, and churn data
- Applied full preprocessing pipeline: one-hot encoding, feature scaling, PCA for dimensionality reduction
- Performed clustering with **K-Means** (k=4) selected via **elbow** and **silhouette analysis**
- Visualized results with PCA scatter plots, boxplots, and stacked bar charts
- Profiled each segment across spend, usage, tenure, and churn risk

**Key Findings:**

<h4>📌 Key Findings</h4>

<table>
  <thead>
    <tr>
      <th>Segment</th>
      <th>Description</th>
      <th>Strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>💬 Voice-Dominant Users</td>
      <td>High voice & intl use,<br>short tenure</td>
      <td>Add voice bundles,<br>retention plans</td>
    </tr>
    <tr>
      <td>📱 High-Usage Streamers</td>
      <td>Heavy data/streaming,<br>higher churn</td>
      <td>Promote unlimited/<br>entertainment perks</td>
    </tr>
    <tr>
      <td>💸 Low-Value Starters</td>
      <td>Low usage,<br>low tenure</td>
      <td>Grow via onboarding<br>& upselling</td>
    </tr>
    <tr>
      <td>🧭 Loyal Minimalists</td>
      <td>Long tenure, low usage,<br>least churn</td>
      <td>Reward loyalty,<br>protect margin</td>
    </tr>
  </tbody>
</table>


**Tech Stack:** `Python`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`  
**Core Skills Demonstrated:** Customer analytics, unsupervised learning, PCA, strategic interpretation, stakeholder communication

👉 [View Full Project](https://pmcavallo.github.io/telecom-segmentation/)  

---
## Customer Churn Predictor

**Goal**: Predict whether a telecom customer is likely to churn using an end-to-end machine learning pipeline.

**Description**:  
This interactive app allows users to input customer features (e.g., tenure, contract type, monthly charges) and receive a real-time churn prediction. It includes data preprocessing, feature engineering, model training, cloud deployment, and live user interaction.

- 🔗 [Live App (Render)](https://churn-prediction-app-dxft.onrender.com/)
- 💻 [GitHub Repo](https://github.com/pmcavallo/churn-prediction-app)
- 📎 Technologies: `Python`, `scikit-learn`, `Streamlit`, `joblib`, `Render`

**Screenshot**:  
![Churn Prediction App Screenshot](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/streamlit_ui.png?raw=true)

#### ⚙️ Tech Stack

| Purpose           | Tool                   |
|-------------------|------------------------|
| Language          | Python 3               |
| ML Library        | scikit-learn           |
| Visualization     | seaborn, matplotlib    |
| Data Handling     | pandas, NumPy          |
| Deployment        | GitHub Pages           |


## 📶 Telecom Engagement Monitoring with Fractional Logistic Regression

This project builds a full monitoring pipeline to track **postpaid customer engagement** over time using simulated telecom data. The model uses **fractional logistic regression** to predict monthly engagement as a proportion and evaluates its stability across development and monitoring datasets.

👉 **[View Full Project Notebook](https://pmcavallo.github.io/engagement-monitoring/)**

---

### 🧰 Tech Stack

| Component            | Library / Tool         |
|---------------------|------------------------|
| Modeling             | `statsmodels` (GLM - Binomial with Logit link) |
| Data Handling        | `pandas`, `numpy`      |
| Evaluation Metrics   | `sklearn.metrics`      |
| Stability Analysis   | Custom PSI logic       |
| Visualization        | `matplotlib`           |

---

### 📌 Highlights & Findings

- **Model Performance Remains Strong**:
  - RMSE and MAE remain consistent across development and monitoring samples.
  - Calibration curves closely track the 45° reference line, confirming that predicted probabilities are well-aligned with observed engagement.

- **Population Stability (PSI) Results**:
  - Most variables, including `engagement_pred`, `age`, and `network_issues`, remained stable (PSI < 0.10).
  - Moderate shifts were observed in `tenure_months` and `avg_monthly_usage`, suggesting slight distributional drift.

- **Final Monitoring Score**:
  - A weighted score combining RMSE delta, MAE delta, and PSI indicated the model is **stable**.
  - ✅ **No immediate action needed**, but moderate PSI shifts warrant ongoing monitoring in future quarters.

- **Vintage-Level Insights**:
  - Predicted and actual engagement increased from **2023Q4 to 2025Q2**, which aligns with expected behavioral trends.

---

This project demonstrates how to proactively monitor engagement models using interpretable statistics and custom stability metrics, with outputs ready for integration into model governance workflows.

### Fraud Detection with XGBoost & SHAP

A simulated end-to-end machine learning pipeline that predicts fraudulent transactions using XGBoost and interprets the model with SHAP values.

#### Objective
Detect fraudulent transactions using synthetic data with engineered features such as transaction type, amount, time, and customer behavior patterns.

#### Key Steps

- **Data Simulation**: Created a synthetic dataset mimicking real-world credit card transactions with class imbalance.
- **Preprocessing**: Handled class imbalance with SMOTE and balanced class weights.
- **Modeling**: Trained an XGBoost classifier and optimized it via grid search.
- **Evaluation**: Evaluated using confusion matrix, ROC AUC, and F1-score.
- **Explainability**: Used SHAP (SHapley Additive exPlanations) to explain model predictions and identify top drivers of fraud.

### ⚙️ Tech Stack

| Purpose           | Tool                   |
|-------------------|------------------------|
| Language          | Python                 |
| ML Library        | XGBoost, scikit-learn  |
| Explainability    | SHAP                   |
| Data Simulation   | NumPy, pandas          |
| Visualization     | matplotlib, seaborn    |
| Deployment        | Local / GitHub         |

#### 📈 Sample Output

- 🔺 Fraud detection accuracy: ~94%
- 🔍 Top features identified by SHAP:
  - `transaction_amount`
  - `time_delta_last_tx`
  - `customer_avg_tx`

📎 [View on GitHub](https://github.com/pmcavallo/fraud-detection-project) 

---

### Airline Flight Delay Prediction with Python

A full machine learning pipeline that predicts flight delays using simulated airline data enriched with real U.S. airport codes and weather features. The project explores exploratory analysis, model training, and practical recommendations for airport operations.

#### Objective
Predict whether a flight will be delayed based on features like carrier, origin, departure time, distance, and simulated weather patterns.

#### Key Steps

- **Data Simulation**: Generated a large synthetic dataset including delay labels and airport metadata.
- **EDA**: Visualized delay patterns by airport, hour of day, and weather impact.
- **Modeling**: Trained a Random Forest classifier with class balancing and hyperparameter tuning.
- **Evaluation**: Assessed performance using confusion matrix, precision-recall, and F1-score.
- **Recommendations**: Delivered operational insights and visualized them with heatmaps and scatterplots.

#### ⚙️ Tech Stack

| Purpose           | Tool                    |
|-------------------|-------------------------|
| Language          | Python 3                |
| ML Library        | scikit-learn            |
| Visualization     | matplotlib, seaborn     |
| Simulation        | NumPy, pandas            |
| Mapping (EDA)     | Plotly, geopandas        |
| Deployment        | GitHub Pages (Markdown) |

#### 📂 Read the Full Report
📎 [View Full Project](https://pmcavallo.github.io/full-airline-delay-project/)

## 🛠️ In Progress

### 🗺️ Geospatial Risk Dashboard (Tableau)
Building an interactive Tableau dashboard to visualize public health and economic risk indicators across Texas counties.

- Skills: `Tableau`, `Data Wrangling`, `Mapping`, `Interactive Filters`

> Will be added soon...

---

## What’s Next
- Migrating model workflows into modular Python scripts
- Adding CI/CD and containerization (e.g., Docker)
- Exploring model monitoring frameworks

---

For more details, view my full [portfolio homepage](https://pmcavallo.github.io) or connect via [LinkedIn](https://www.linkedin.com/in/paulocavallo/).
