---
layout: default
title: 🚀 Projects
permalink: /projects/
---

A selection of hands-on projects demonstrating real-world data science, modeling, and cloud deployment — designed and built using Python, scikit-learn, and modern tools like Streamlit and Render.

---

## 📊 Customer Churn Predictor

**Goal**: Predict whether a telecom customer is likely to churn using an end-to-end machine learning pipeline.

**Description**:  
This interactive app allows users to input customer features (e.g., tenure, contract type, monthly charges) and receive a real-time churn prediction. It includes data preprocessing, feature engineering, model training, cloud deployment, and live user interaction.

- 🔗 [Live App (Render)](https://churn-prediction-app-dxft.onrender.com/)
- 💻 [GitHub Repo](https://github.com/pmcavallo/churn-prediction-app)
- 📎 Technologies: `Python`, `scikit-learn`, `Streamlit`, `joblib`, `Render`

**Screenshot**:  
![Churn Prediction App Screenshot](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/streamlit_ui.png?raw=true)

#### ⚙️ Tech Stack

<table>
  <tr><th>Purpose</th><th>Tool(s)</th></tr>
  <tr><td>Language</td><td><strong>Python 3</strong></td></tr>
  <tr><td>ML Library</td><td>XGBoost, scikit-learn</td></tr>
  <tr><td>Explainability</td><td>SHAP</td></tr>
  <tr><td>Data Simulation</td><td>NumPy, pandas</td></tr>
  <tr><td>Visualization</td><td>matplotlib, seaborn</td></tr>
  <tr><td>Deployment</td><td>Local / GitHub</td></tr>
</table>


### 🕵️‍♂️ Fraud Detection with XGBoost & SHAP

A simulated end-to-end machine learning pipeline that predicts fraudulent transactions using XGBoost and interprets the model with SHAP values.

#### 🔍 Objective
Detect fraudulent transactions using synthetic data with engineered features such as transaction type, amount, time, and customer behavior patterns.

#### 🧠 Key Steps

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

### ✈️ Airline Flight Delay Prediction with Python

A full machine learning pipeline that predicts flight delays using simulated airline data enriched with real U.S. airport codes and weather features. The project explores exploratory analysis, model training, and practical recommendations for airport operations.

#### 🔍 Objective
Predict whether a flight will be delayed based on features like carrier, origin, departure time, distance, and simulated weather patterns.

#### 🧠 Key Steps

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

## 🧵 What’s Next
- Migrating model workflows into modular Python scripts
- Adding CI/CD and containerization (e.g., Docker)
- Exploring model monitoring frameworks

---

For more details, view my full [portfolio homepage](https://pmcavallo.github.io) or connect via [LinkedIn](https://www.linkedin.com/in/paulocavallo/).
