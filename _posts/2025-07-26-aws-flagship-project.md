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

```

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
📊 Model Performance Summary

| Metric       | Value  |
| ------------ | ------ |
| AUC (ROC)    | \~0.49 |
| KS Statistic | \~0.03 |

⚙️ Interpretation
- Model performance is poor (intentionally left unoptimized) — validating Spark integration, not predictive accuracy.
- This section of the project was not focused on achieving optimal model performance but rather on showcasing the integration of PySpark for scalable ETL and modeling workflows.
- The AUC (~0.49) and KS (~0.03) reflect a model performing no better than random.
- The synthetic dataset was intentionally simplified and lacks:
  - Deep feature engineering
  - Domain-calibrated signals
  - Real-world complexity (e.g., bureau data, macroeconomic indicators, payment history)
- Parquet outputs are production-friendly and portable (e.g., for AWS S3 ingestion).

---

## 3. AWS S3 Upload (via CLI)

```bash
aws s3 cp /mnt/data/credit_flagship.parquet s3://aws-flagship-project/credit-risk/cleaned/credit_flagship.parquet --recursive
```

---

## 4. Modeling with CatBoost

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
X = df.drop(columns='loan_status_flag')
y = df['loan_status_flag']

X = pd.get_dummies(X, columns=['state', 'util_bin'], drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, cat_features=[], verbose=0)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_val)[:, 1]
print(f"AUC: {roc_auc_score(y_val, y_pred_proba):.3f}")
```

---

## 5. SHAP Interpretability

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val)
```

Key insights:
- **FICO score** is the strongest driver; higher scores sharply reduce risk.
- **Utilization and CLTV** also contribute to higher risk.
- **Delinquency flag** is binary but adds strong predictive signal.

---

## 6. Segment-Level Analysis by State

```python
X_val_copy = X_val.copy()
X_val_copy['prediction'] = y_pred_proba
X_val_copy['actual'] = y_val.values

X_val_copy['state'] = df.loc[X_val_copy.index, 'state']

segment_df = X_val_copy.groupby('state').agg(
    avg_predicted=('prediction', 'mean'),
    actual_rate=('actual', 'mean')
).sort_values('actual_rate', ascending=False)

segment_df.plot(kind='barh', figsize=(8, 5), title="CatBoost – Predicted vs Actual Default Rate by State")
```

---

## 7. Business Recommendation

- **Tailored Thresholds**: Adjust risk thresholds based on state-level accuracy gaps. NY and FL underperform slightly in predicted rates—consider model recalibration.
- **Scorecard Enhancement**: FICO score dominates the model. Future iterations should regularize this influence or augment with behavior-based features.
- **S3 Data Pipelines**: Spark → Parquet → S3 integration is clean. Automate using **Airflow** in production.
- **Explainability-Ready**: SHAP makes the model audit-friendly. Aligns with regulatory needs (SR 11-7 or CCAR-style audits).
- **Next Steps**:
  - Extend to LGD modeling using fractional logistic regression.
  - Incorporate **Snowflake** as centralized data source.
  - Productionalize with **SageMaker Pipelines** or **Lambda triggers** from S3.

---

_Last updated: 2025-07-26 20:38_
