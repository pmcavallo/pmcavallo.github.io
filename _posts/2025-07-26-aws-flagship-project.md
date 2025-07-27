---
layout: post
title: 💳 Credit Risk Model Deployment & Monitoring (AWS + PySpark + CatBoost)
--- 

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

I simulate a synthetic telecom-style credit dataset with known risk drivers and regional variation.

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
*⚠️ Note: Synthetic logic only: Flags are generated from loose rules to simulate borrower risk — not fitted on real defaults.*

---

## 2. PySpark ETL and Preprocessing

I use PySpark for scalable preprocessing and feature engineering of raw credit data. This simulates a real-world scenario where data pipelines need to handle large volumes efficiently before feeding downstream ML services like AWS SageMaker.

```python
print("🚀 Script has started running...")

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, when

# Initialize Spark session
spark = SparkSession.builder.appName("Credit Risk ETL Updated").getOrCreate()

# Define schema explicitly
schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("fico_score", DoubleType(), True),
    StructField("loan_amount", DoubleType(), True),
    StructField("tenure_months", IntegerType(), True),
    StructField("state", StringType(), True),
    StructField("plan_type", StringType(), True),
    StructField("monthly_income", DoubleType(), True),
    StructField("date_issued", StringType(), True),
    StructField("loan_status", StringType(), True),
    StructField("loan_status_flag", IntegerType(), True),
    StructField("credit_utilization", DoubleType(), True),
    StructField("has_bankruptcy", IntegerType(), True)
])

# Load data
df = spark.read.csv("C:/credit_risk_project/data/raw/credit_data_aws_flagship2.csv", schema=schema, header=True)
print(f"✅ Loaded {df.count()} rows")


# Feature engineering
df = df.withColumn("loan_status_flag", col("loan_status_flag").cast("int"))

df = df.withColumn("cltv", (col("monthly_income") * col("tenure_months") / 12) - col("loan_amount"))

df = df.withColumn("util_bin", when(col("credit_utilization") < 0.3, "Low")
                                   .when(col("credit_utilization") < 0.6, "Medium")
                                   .otherwise("High"))

df = df.withColumn("delinq_flag", (col("loan_status_flag") == 1).cast("int"))

df = df.withColumn("high_risk_flag", when((col("fico_score") < 580) |
                                          (col("plan_type") == "Business") |
                                          (col("has_bankruptcy") == 1), 1).otherwise(0))

# Optional: preview
df.select("customer_id", "fico_score", "loan_amount", "cltv", "util_bin", "delinq_flag", "high_risk_flag").show(10)

# Drop rows with missing target or important fields
df_cleaned = df.dropna(subset=["fico_score", "loan_amount", "monthly_income", "loan_status_flag"])

df.groupBy("loan_status_flag").count().show()
df.select("fico_score").filter("fico_score IS NULL").count()

# ✅ Save full cleaned dataset to new Parquet file
df_cleaned.write.mode("overwrite").parquet("output/credit_data_cleaned2.parquet")

# Stop Spark session
spark.stop()
print("✅ ETL completed.")

```
![Spark](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/spark2.png?raw=true) 

✅ Highlights:
- Derived variables: cltv, util_bin, delinq_flag, high_risk_flag
- Designed for high scalability (Spark)
- Output stored in Parquet format for downstream consumption

🔁 Logistic Regression with PySpark MLlib
This step validates the ability to train and evaluate a logistic regression model in a distributed Spark environment — with class weights, regularization, and KS evaluation logic.

```python
"""
Logistic Regression with Class Weights, Regularization, and KS Evaluation
For Credit Risk Modeling using PySpark
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Start Spark session with increased memory
spark = SparkSession.builder \
    .appName("CreditRiskLogisticRegressionFinal") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
    .getOrCreate()

# Reduce log level
spark.sparkContext.setLogLevel("ERROR")


print("🚀 Script has started running...")
print("✅ Spark session started.")

# Load the transformed data (adjust path if needed)
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")
print("✅ Data split done.")

# STEP 1: Create class weights
label_counts = df.groupBy("loan_status_flag").count().toPandas()
majority = label_counts["count"].max()
weights = {
    int(row["loan_status_flag"]): majority / row["count"]
    for _, row in label_counts.iterrows()
}

from pyspark.sql.functions import when

df = df.withColumn(
    "classWeightCol",
    when(col("loan_status_flag") == 1, float(weights[1]))
    .otherwise(float(weights[0]))
)
print(f"✅ Added classWeightCol. Weights: {weights}")

# STEP 2: Assemble features manually
#df.printSchema()
features = [
    'fico_score',
    'loan_amount',
    'tenure_months',
    'monthly_income',
    'credit_utilization',
    'has_bankruptcy',
    'cltv',
    'high_risk_flag'
]

assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df).select("features", "loan_status_flag", "classWeightCol")

# STEP 3: Logistic regression with regularization
lr = LogisticRegression(
    labelCol="loan_status_flag",
    featuresCol="features",
    weightCol="classWeightCol",
    regParam=0.1,
    elasticNetParam=0.0  # L2 only
)

# STEP 4: Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print("✅ Data split done.")  # Count logging removed to prevent Java gateway crash

# STEP 5: Fit model
model = lr.fit(train_df)
predictions = model.transform(test_df)

# STEP 6: Evaluate AUC
evaluator = BinaryClassificationEvaluator(
    labelCol="loan_status_flag",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print(f"✅ Logistic Regression AUC: {auc:.4f}")

# STEP 7: KS calculation
# Get prediction probabilities and true labels
from pyspark.ml.functions import vector_to_array

pred_df = predictions \
    .withColumn("probability_array", vector_to_array("probability")) \
    .withColumn("prob_default", col("probability_array")[1]) \
    .select("loan_status_flag", "prob_default")


# Create deciles
from pyspark.sql.window import Window
from pyspark.sql.functions import ntile, sum as spark_sum

windowSpec = Window.orderBy(col("prob_default").desc())
ks_df = pred_df.withColumn("decile", ntile(10).over(windowSpec))

# Aggregate by decile
agg_df = ks_df.groupBy("decile").agg(
    spark_sum((col("loan_status_flag") == 1).cast("int")).alias("bads"),
    spark_sum((col("loan_status_flag") == 0).cast("int")).alias("goods")
).orderBy("decile")

# Calculate cumulative bads/goods and KS
from pyspark.sql.functions import lit

total = agg_df.selectExpr("sum(bads) as total_bads", "sum(goods) as total_goods").collect()[0]
total_bads = total["total_bads"]
total_goods = total["total_goods"]

agg_df = agg_df.withColumn("cum_bads", spark_sum("bads").over(Window.orderBy("decile")))
agg_df = agg_df.withColumn("cum_goods", spark_sum("goods").over(Window.orderBy("decile")))
agg_df = agg_df.withColumn("cum_bad_pct", col("cum_bads") / lit(total_bads))
agg_df = agg_df.withColumn("cum_good_pct", col("cum_goods") / lit(total_goods))
agg_df = agg_df.withColumn("ks", (col("cum_bad_pct") - col("cum_good_pct")).cast("double"))

ks_value = agg_df.agg({"ks": "max"}).collect()[0][0]
print(f"✅ KS Statistic: {ks_value:.4f}")

spark.stop()
print("✅ Spark session stopped.")
```
![Spark](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/spark3.png?raw=true) 

📊 Model Performance Summary
The logistic regression model trained using Spark produced the following metrics:
✅ AUC (Area Under the ROC Curve): 0.4931.
✅ KS (Kolmogorov-Smirnov) Statistic: 0.0342.

⚙️ Interpretation
- The AUC (~0.49) and KS (~0.03) reflect a model performing no better than random.
- This weak performance is expected at this stage due to:
  - Lack of advanced feature transformations
  - Imbalanced class distribution
  - Linear model limitations on complex patterns
- The synthetic dataset was intentionally simplified and lacks:
  - Deep feature engineering
  - Domain-calibrated signals
  - Real-world complexity (e.g., bureau data, macroeconomic indicators, payment history)

This section of the project was not focused on achieving optimal model performance but rather on showcasing the integration of PySpark for scalable ETL and modeling workflows. It produces a final Parquet file, wihch are production-friendly and portable (e.g., for AWS S3 ingestion). The next phase will transition to Python-based modeling, where feature engineering and model tuning can be more flexibly applied to improve performance.

* Note: ⚠️ This logistic regression is included to demonstrate ML integration in a Spark pipeline. Its performance is not expected to exceed random, given intentionally limited features.*
* 
---

## 3. AWS S3 Upload (via CLI)

```bash
aws s3 cp /mnt/data/credit_flagship.parquet s3://aws-flagship-project/credit-risk/cleaned/credit_flagship.parquet --recursive
```

---

## 4. Modeling with CatBoost

Even in an ML engineering–oriented project like this, we should never skip EDA or single-factor checks, because:

| 🧠 Reason                               | 💡 Why It Matters                                                  |
|----------------------------------------|--------------------------------------------------------------------|
| **Understand feature distributions**       | Know which variables are usable, sparse, or redundant             |
| 🔍 **Identify data leakage**               | Some features (e.g., flags) might encode the target               |
| 💡 **Get modeling insights**               | E.g., monotonicity, expected sign, binning candidates             |
| ⚖️ **Select features and transformations** | Before blindly throwing everything into a model                   |
| 📚 **Supports SHAP and explainability**    | Helps confirm whether SHAP makes sense                            |


```python
df['high_risk_flag'].value_counts(normalize=True)

# Categorical univariate breakdown
for col in ['plan_type', 'state', 'util_bin']:
    print(f"\n=== {col.upper()} ===")
    print(df.groupby(col)['high_risk_flag'].mean().sort_values(ascending=False))

for col in ['has_bankruptcy', 'delinq_flag']:
    print(f"{col}:")
    print(df.groupby(col)['high_risk_flag'].mean(), "\n")

for col in ['fico_score', 'loan_amount', 'monthly_income', 'credit_utilization']:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x='high_risk_flag', y=col, data=df)
    plt.title(f"{col} vs. high_risk_flag")
    plt.tight_layout()
    plt.show()
```

Business plans show the highest risk with a 100% high-risk flag rate, while Individual and Family plans have much lower rates (~11–12%). Among states, NY and WA exhibit the highest risk (~42%), while FL has the lowest (~33%). Higher credit utilization correlates with slightly higher risk, though the difference across buckets is modest. Customers with prior bankruptcies show a significantly higher risk (41.5%) compared to those without (30.2%), and delinquent borrowers are also more likely to be flagged high risk (41.5% vs. 36.7%).

![Spark](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/spark4.png?raw=true) 
![Spark](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/spark5.png?raw=true) 

✅ Interpretation of Single-Factor Boxplots vs high_risk_flag

| Feature            | Observation                          | Interpretation                              |
|--------------------|--------------------------------------|----------------------------------------------|
| **fico_score**     | Very slight dip for 1                | Not a strong separator — useful, but not dominant |
| **loan_amount**    | Overlapping distributions            | Not useful alone, but may interact with others |
| **monthly_income** | Slightly lower for 1                 | Weak signal — no leakage                      |
| **credit_utilization** | Slight right shift for 1         | Some discriminatory power — useful           |


## Modeling with CatBoost

I chose CatBoost for this credit risk project because it balances model performance with explainability, supports categorical features natively, and and is gaining traction in regulated industries.

```python
!pip install catboost s3fs --quiet

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import boto3
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

s3_path = 's3://aws-flagship-project/credit-risk/cleaned/'
df = pd.read_parquet(s3_path, engine='pyarrow')
```

```python
# Target
target = 'high_risk_flag'

# Drop leaking features
leak_features = ['has_bankruptcy', 'plan_type', 'customer_id', 'date_issued', 'loan_status']

# Filter final feature set
df_model = df.drop(columns=leak_features)

# Split features and target
X = df_model.drop(columns=[target])
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Predict probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")

# KS
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ks = max(tpr - fpr)
print(f"KS: {ks:.4f}")
```

## 5. Feature Importance and SHAP Interpretability

Here I use SHAP to ensure the model's predictions are driven by meaningful risk factors — not proxy variables or data artifacts — and that feature importance aligns with domain expectations.

```python
# Convert cat feature names to indices (required if using column names throws errors)
cat_feature_indices = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]

# Get importance values safely
importances = model.get_feature_importance(Pool(X_train, label=y_train, cat_features=cat_feature_indices))

# Build and plot
feature_names = X_train.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
feat_imp.head(15).plot(kind='barh')
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

```
![Spark](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/spark6.png?raw=true) 

---

```python
!pip install shap --quiet
import shap

# Initialize JS rendering for Jupyter (force plot compatibility)
shap.initjs()

# Use TreeExplainer for CatBoost
explainer = shap.TreeExplainer(model)

# Compute SHAP values for test set
shap_values = explainer.shap_values(X_test)

# Summary plot (beeswarm)
shap.summary_plot(shap_values, X_test)
```
![Spark](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/spark7.png?raw=true) 

Key insights:
- **FICO score** is the strongest driver; higher scores sharply reduce risk.
- **Utilization and CLTV** also contribute to higher risk.
- **Delinquency flag** is binary but adds strong predictive signal.

---

## 6. Segment-Level Analysis by State

```python
# Add predictions to training set
X_seg = X_train.copy()
X_seg['prediction'] = model.predict_proba(X_train)[:, 1]
X_seg['actual'] = y_train.values

# Group by state
segment_df = X_seg.groupby('state').agg(
    avg_predicted = ('prediction', 'mean'),
    actual_rate = ('actual', 'mean'),
    count = ('actual', 'count')
)

# Optional: AUC per state (only if both classes are present)
def state_auc(group):
    if group['actual'].nunique() == 2:
        return roc_auc_score(group['actual'], group['prediction'])
    return np.nan

segment_df['auc'] = X_seg.groupby('state').apply(state_auc)

# Over/underprediction
segment_df['overprediction'] = segment_df['avg_predicted'] - segment_df['actual_rate']

# Bar chart
segment_df[['avg_predicted', 'actual_rate']].sort_values('actual_rate').plot(
    kind='barh', figsize=(8,6))
plt.title("CatBoost – Predicted vs Actual Default Rate by State")
plt.xlabel("Rate")
plt.tight_layout()
plt.show()
```
![Spark](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/spark8.png?raw=true) 

🔍 Interpretation of the Segment Plot

This bar chart compares the average predicted default probability (blue) with the actual observed rate (orange) by state:
- Overprediction: In states like FL, the model consistently overestimates risk.
- Underprediction: In NY, the model underpredicts risk — predicted probabilities are lower than actual rates.
- Tight alignment: For states like CA, IL, TX, the model tracks reality closely.

---

## 7. Business Recommendation

- **State-Level Threshold Calibration**: Adjust risk cutoffs where predicted risk diverges from observed default rates (e.g., FL and NY).
- **Scorecard Enhancement**: FICO score dominates the model. Future iterations should regularize this influence or augment with behavior-based features.
- **S3 Data Pipelines**: Spark → Parquet → S3 integration is clean. Automate using **Airflow** in production.
- **Next Steps**:
  - Incorporate **Snowflake** as centralized data source.
  - Productionalize with **SageMaker Pipelines** or **Lambda triggers** from S3.

This project simulates a production-ready credit risk pipeline — designed for scalability, interpretability, and strategic alignment.

---

