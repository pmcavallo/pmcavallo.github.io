---
layout: post
title: Airline Flight Delay Prediction with Python
---

This project aims to predict whether a flight will be significantly delayed (15+ minutes) using flight metadata,
weather, and carrier information. Understanding delay drivers is essential for airlines and airports to improve
operations and passenger experience.

I use a simulated dataset with 50,000 rows featuring fake airlines and real U.S. airport codes.

---

## 2. Data Loading and Inspection

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate a large airline dataset
np.random.seed(42)
n = 50000
airport_codes = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MCO']
airlines = ['SkyJet', 'AeroBlue', 'NimbusAir', 'AltusWings', 'FalconExpress']

df = pd.DataFrame({
    'flight_id': np.arange(1, n + 1),
    'airline': np.random.choice(airlines, n),
    'origin': np.random.choice(airport_codes, n),
    'destination': np.random.choice(airport_codes, n),
    'scheduled_departure': np.random.randint(0, 24, n),
    'departure_delay': np.random.normal(loc=10, scale=20, size=n).astype(int),
    'flight_duration': np.random.randint(60, 360, n),
    'distance_miles': np.random.randint(200, 3000, n),
    'weather_delay': np.random.binomial(1, 0.1, size=n),
    'carrier_delay': np.random.binomial(1, 0.15, size=n),
})
df['departure_delay'] = df['departure_delay'].apply(lambda x: max(-15, x))
df['is_delayed'] = (df['departure_delay'] > 15).astype(int)

print(df.head())
print(df.describe(include='all'))
```

---

## 3. Exploratory Data Analysis (EDA)

```python
sns.countplot(x='is_delayed', data=df)
plt.title('Flight Delay Distribution')
plt.show()

sns.boxplot(x='airline', y='departure_delay', data=df)
plt.xticks(rotation=45)
plt.title('Delays by Airline')
plt.show()

sns.lineplot(x='scheduled_departure', y='departure_delay', data=df)
plt.title('Average Delay by Hour of Day')
plt.show()
```
![Flight Delay Distribution](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/airline.png?raw=true) 

This bar chart shows the overall balance between on-time and significantly delayed flights in the dataset. We typically see a much higher count of on-time flights, which confirms a class imbalance problem.

![Delays by Airline](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/airline2.png?raw=true) 

This plot helps visualize how different airlines vary in their delay patterns. Airlines with taller boxplots or more extreme outliers tend to have greater variability and potentially higher average delays. If certain airlines consistently show higher median delays, this might indicate operational inefficiencies or scheduling issues that could be modeled more explicitly.

![Average Delay bu Hour of Day](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/airline3.png?raw=true) 

This time-based trend reveals how delays fluctuate throughout the day. It's common to observe a build-up of delays in the afternoon and evening, due to cumulative effects from earlier flights and congested air traffic. Flights scheduled later in the day may have higher delay risk, which suggests that scheduled_departure is a valuable predictive feature and could be binned or transformed for better modeling.

---

## 4. Data Preprocessing

```python
X = df.drop(columns=['flight_id', 'departure_delay', 'is_delayed'])
y = df['is_delayed']
X_encoded = pd.get_dummies(X, drop_first=True)
```

---

## 5. Model Building

Here, I'm going to use a random forest classifier. Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction accuracy and control overfitting. It can be used for both classification and regression tasks. 

How it works:
- Bootstrap Sampling: The algorithm creates many subsets of the training data using bootstrapping (sampling with replacement).
- Grow Decision Trees: A decision tree is trained on each subset, but at each split, only a random subset of features is considered (not all).
- Voting or Averaging:
  - For classification, each tree votes for a class, and the majority vote is the final prediction.
  - For regression, the average of all trees' predictions is the output.


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42) #the model will build 100 decision trees
model.fit(X_train, y_train)
```

---

## 6. Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()
```
![Results](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/airline_results.png?raw=true)
---

## 7. Interpretation


- Most important features: `distance_miles`, `flight_duration`, and `scheduled_departure`.

---

## 8. Detailed Model Interpretation

### Original Confusion Matrix

```
[[5114 1069]
 [3197  620]]
```

| Class         | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| On-Time (0)   | 61.5%     | 83.0%  | 70.7%    |
| Delayed (1)   | 36.7%     | 16.2%  | 22.6%    |
| **Accuracy**  |           |        | 57.3%    |

- The model misses over 80% of delays.
- High precision but low recall means it's conservative in predicting delays.

---

## 9. First Adjustment: Class Weight Balancing
Class weight balancing is a technique used in classification models to handle imbalanced datasets by assigning higher importance (or “weight”) to the minority class during training. When one class (e.g., “delayed flights”) has far fewer examples than the other (e.g., “on-time flights”), the model may become biased and predict the majority class more often simply to maximize accuracy. By setting class_weight='balanced', the algorithm automatically adjusts the penalty for misclassifying each class based on their frequency—penalizing errors on the minority class more heavily. This encourages the model to pay more attention to the underrepresented class, helping improve recall and reducing the risk of overlooking critical cases.

```python
model_weighted = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)

print(confusion_matrix(y_test, y_pred_weighted))
print(classification_report(y_test, y_pred_weighted))
```

### Updated Confusion Matrix

```
[[5205  978]
 [3252  565]]
```

| Class         | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| On-Time (0)   | 61.5%     | 84.2%  | 71.0%    |
| Delayed (1)   | 36.6%     | 14.8%  | 21.1%    |
| **Accuracy**  |           |        | 57.7%    |

- Class weight balancing slightly improved coverage of delays but reduced overall stability.
- Still very low recall for delayed class.

---

## 10. Recommendations & Next Steps

### Immediate Actions
- ✅ Lower classification threshold from 0.5 to ~0.3
- ✅ Test `class_weight='balanced'`
- ✅ Visualize precision-recall tradeoff

### Next Improvements
- 🧪 Use SMOTE oversampling to balance training data
- 🔍 Add features: route-level averages, aircraft delay history, day-of-week
- 🔄 Try alternate models like Logistic Regression or XGBoost
- 📊 Tune hyperparameters and threshold together

### Deployment Advice
- If business goal is **delay prevention**, prioritize **recall**
- Use this as a **screening tool** to escalate flight risk checks


---

## 11. SMOTE: Synthetic Minority Oversampling

To address class imbalance and improve recall, I apply SMOTE (Synthetic Minority Oversampling Technique) to upsample the delayed flights during training. Instead of simply duplicating minority class samples, SMOTE generates synthetic examples by interpolating between existing minority instances and their nearest neighbors. This helps the model learn more generalizable decision boundaries without overfitting to repeated data. By balancing the number of examples in each class, SMOTE improves the model’s ability to recognize underrepresented outcomes—such as flight delays or fraud cases—leading to better recall and more reliable performance in real-world applications where imbalanced data is common.

### Code: Apply SMOTE

```python
!pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report

# Encode and split data
X = df_airline.drop(columns=['flight_id', 'departure_delay', 'is_delayed'])
y = df_airline['is_delayed']
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 12. SMOTE Results & Interpretation

### Confusion Matrix

```
[[4539 1644]
 [2835  982]]
```

| Class         | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| On-Time (0)   | 61.6%     | 73.4%  | 67.0%    |
| Delayed (1)   | 37.4%     | 25.7%  | 30.4%    |
| **Accuracy**  |           |        | 55.2%    |

### Observations

- 🔼 **Recall for delayed flights improved to 25.7%** (from 14.8%)
- 🔼 **F1-score for delays increased to 30.4%** (from 21.1%)
- 🔽 Slight decrease in overall accuracy
- ✅ This is a good tradeoff when the goal is **catching more delays** (even with more false positives)

---

## 13. Final Recommendations

- **If recall is the top priority**, SMOTE clearly outperforms the other strategies so far.
- We might consider using SMOTE together with **threshold tuning** for even better results.
- For production deployment, explore:
  - 🚀 XGBoost or Logistic Regression with ROC analysis
  - ⏳ Time-based features (prior flight delays, rolling averages)
  - 📡 Live weather and congestion feeds

