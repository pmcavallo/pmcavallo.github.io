---
layout: post
title: 📊 Customer Segmentation Using Statistical Clustering
---
This project applies **statistical segmentation techniques** to a simulated postpaid telecom customer base in order to uncover distinct user personas. This segmentation model can be used to support personalized retention strategies, plan design, and marketing optimization.

---

## 📁 Dataset Overview (Simulated)
I simulate 5,000 postpaid customers with realistic behavior patterns across multiple usage and demographic dimensions.

**Features include:**
- `monthly_charge`: Monthly billing amount
- `data_usage_gb`: Monthly data usage in GB
- `call_minutes`: Monthly voice call usage
- `intl_calls`: Count of international calls
- `streaming_usage`: Estimated monthly streaming (hrs)
- `device_type`: Smartphone, Tablet, Hotspot
- `contract_type`: Month-to-month, 1 year, 2 year
- `tenure_months`: Customer lifetime in months
- `support_calls`: Number of customer support calls
- `payment_method`: Auto-pay or Manual
- `churn`: Binary indicator (simulated) for churned accounts

```python
# Data simulation
import numpy as np
import pandas as pd

np.random.seed(42)
n_customers = 5000

monthly_charge = np.round(np.random.normal(80, 25, n_customers).clip(20, 200), 2)
data_usage_gb = np.round(np.random.gamma(shape=2, scale=3, size=n_customers), 2)
call_minutes = np.round(np.random.normal(500, 150, n_customers).clip(50, 1500), 0)
intl_calls = np.random.poisson(lam=0.5, size=n_customers)
streaming_usage = np.round(data_usage_gb * np.random.uniform(0.4, 1.2, size=n_customers), 2)
device_type = np.random.choice(['Smartphone', 'Tablet', 'Hotspot'], size=n_customers, p=[0.8, 0.15, 0.05])
contract_type = np.random.choice(['Month-to-month', '1 year', '2 year'], size=n_customers, p=[0.5, 0.3, 0.2])
tenure_months = np.round(np.random.exponential(scale=24, size=n_customers)).astype(int).clip(1, 72)
support_calls = np.random.poisson(lam=1.2, size=n_customers).clip(0, 10)
payment_method = np.random.choice(['Auto-pay', 'Manual'], size=n_customers, p=[0.7, 0.3])

churn = np.where((contract_type == 'Month-to-month') & (support_calls > 2) & (tenure_months < 12),
                 np.random.binomial(1, 0.4, n_customers),
                 np.random.binomial(1, 0.1, n_customers))

df = pd.DataFrame({
    'monthly_charge': monthly_charge,
    'data_usage_gb': data_usage_gb,
    'call_minutes': call_minutes,
    'intl_calls': intl_calls,
    'streaming_usage': streaming_usage,
    'device_type': device_type,
    'contract_type': contract_type,
    'tenure_months': tenure_months,
    'support_calls': support_calls,
    'payment_method': payment_method,
    'churn': churn
})
```

---

## 🔧 Preprocessing
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

features = df.drop(columns=['churn'])
numeric_features = ['monthly_charge', 'data_usage_gb', 'call_minutes', 'intl_calls',
                    'streaming_usage', 'tenure_months', 'support_calls']
categorical_features = ['device_type', 'contract_type', 'payment_method']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

X_preprocessed = preprocessor.fit_transform(features)
encoded_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
final_feature_names = numeric_features + list(encoded_cat_cols)

X_df = pd.DataFrame(X_preprocessed.toarray() if hasattr(X_preprocessed, "toarray") else X_preprocessed,
                    columns=final_feature_names)
```

---

## 📉 PCA for Dimensionality Reduction
**Principal Component Analysis (PCA)** projects high-dimensional data into orthogonal dimensions that capture the most variance.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_df)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.5)
plt.title("PCA Projection of Customers")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()
```
![PCA](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg1.png?raw=true) 

Explanation: The PCA scatterplot helps visualize how customers are distributed by the top two components. PC1 explains 24% and PC2 about 13% of the variance.

---

## 🔍 K-Means Clustering & Elbow Method
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_df)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_df, kmeans.labels_))

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method: Inertia vs. K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()
```
![elbow method](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg2.png?raw=true) 
Explanation: The elbow method shows diminishing gains after k=4, while silhouette scores drop sharply after 2. We choose **k=4**.

---

## 🧭 Final Clustering and PCA Visualization

```python
# Final KMeans model with k=4
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_df)
pca_df['cluster'] = cluster_labels

# Visualize final clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='Set2', alpha=0.7)
plt.title("K-Means Clustering (k = 4) in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
```
![cluster visualization](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg3.png?raw=true) 

Explanation: This scatterplot shows how clusters are distributed in PCA space. Cluster 1 (orange) is distinctly separated, indicating high behavioral contrast.

---

## 📊 Segment Profiling

```python
# Attach cluster labels
df['cluster'] = cluster_labels

# Profile by numeric mean
cluster_profile = df.groupby('cluster').agg({
    'monthly_charge': 'mean',
    'data_usage_gb': 'mean',
    'call_minutes': 'mean',
    'intl_calls': 'mean',
    'streaming_usage': 'mean',
    'tenure_months': 'mean',
    'support_calls': 'mean',
    'churn': 'mean'
}).round(2)

# Visual display (in notebook this would be styled)
print(cluster_profile)
```
![cluster](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg4.png?raw=true) 

Explanation: Cluster 1 has the highest data and streaming. Cluster 3 has the longest tenure and lowest churn.

---

## 📈 Visual Analysis

### 📦 Monthly Charge by Cluster

```python
sns.boxplot(data=df, x='cluster', y='monthly_charge', palette='Set2')
plt.title("Monthly Charge by Customer Segment")
plt.xlabel("Cluster")
plt.ylabel("Monthly Charge ($)")
plt.tight_layout()
plt.show()
```
![PCA](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg5.png?raw=true) 

### 📶 Data Usage

```python
sns.boxplot(data=df, x='cluster', y='data_usage_gb', palette='Set2')
plt.title("Data Usage (GB) by Customer Segment")
plt.xlabel("Cluster")
plt.ylabel("Monthly Data Usage (GB)")
plt.tight_layout()
plt.show()
```
![PCA](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg7.png?raw=true) 

### 📺 Streaming Usage

```python
sns.boxplot(data=df, x='cluster', y='streaming_usage', palette='Set2')
plt.title("Streaming Usage by Customer Segment")
plt.xlabel("Cluster")
plt.ylabel("Monthly Streaming Hours")
plt.tight_layout()
plt.show()
```
![PCA](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg8.png?raw=true) 

### ☎️ Support Calls

```python
sns.boxplot(data=df, x='cluster', y='support_calls', palette='Set2')
plt.title("Support Calls by Customer Segment")
plt.xlabel("Cluster")
plt.ylabel("Number of Support Calls")
plt.tight_layout()
plt.show()
```
![PCA](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg6.png?raw=true) 

### ⚠️ Churn Rate by Cluster

```python
sns.barplot(data=df, x='cluster', y='churn', palette='Set2')
plt.title("Churn Rate by Customer Segment")
plt.xlabel("Cluster")
plt.ylabel("Churn Rate")
plt.ylim(0, 0.2)
plt.tight_layout()
plt.show()
```
![PCA](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg9.png?raw=true) 
---

## 📊 Contract & Payment Breakdown

### Contract Type

```python
contract_counts = df.groupby(['cluster', 'contract_type']).size().unstack().fillna(0)
contract_props = (contract_counts.T / contract_counts.sum(axis=1)).T

contract_props.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Set2')
plt.title("Contract Type Distribution by Segment")
plt.xlabel("Cluster")
plt.ylabel("Proportion")
plt.legend(title="Contract Type")
plt.tight_layout()
plt.show()
```
![PCA](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg10.png?raw=true) 
### Payment Method

```python
pay_counts = df.groupby(['cluster', 'payment_method']).size().unstack().fillna(0)
pay_props = (pay_counts.T / pay_counts.sum(axis=1)).T

pay_props.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Set2')
plt.title("Payment Method Distribution by Segment")
plt.xlabel("Cluster")
plt.ylabel("Proportion")
plt.legend(title="Payment Method")
plt.tight_layout()
plt.show()
```
![PCA](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/seg11.png?raw=true) 
---

### 🏷️ Segment Interpretation & Strategy

| Cluster | Label                   | Characteristics                                | Strategy                                                                |
|--------:|-------------------------|------------------------------------------------|-------------------------------------------------------------------------|
| 0       | 💬 Voice-Dominant Users | High intl/voice, low data, short tenure        | Offer voice bundle or international call upgrades                      |
| 1       | 📱 High-Usage Streamers | Highest data and streaming, moderate churn     | Promote unlimited plans or entertainment perks                         |
| 2       | 💸 Low-Value Starters   | Low usage, low tenure, low spend               | Nurture via onboarding, budget-friendly data boosters                  |
| 3       | 🧭 Loyal Minimalists    | Long tenure, low usage, lowest churn           | Reward loyalty, upsell on price-sensitive or family bundle offers      |


---

## ✅ Conclusion

This project demonstrates:
- End-to-end unsupervised modeling using real-world techniques
- Clear communication of data insights and business strategy
- Scalable code and methodology for application in churn modeling, targeting, and marketing

*Built with Python, pandas, scikit-learn, seaborn, and matplotlib.*
