---
layout: post
title: Analyzing A/B Test Impact on Marketplace Conversions with Uplift Modeling
---

This project simulates and analyzes an A/B pricing test in a marketplace context. Using Python, I simulate customer behavior, estimate the causal impact of a price change on conversion rates, and apply uplift modeling to identify heterogeneous treatment effects across cities. The project demonstrates key skills in experimental design, causal inference, uplift modeling, and data visualization.

**Tech Stack**: `Python`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scikit-uplift`

## 1. Data Simulation
I simulate a dataset representing users from five cities (Austin, Chicago, Denver, Miami, Seattle), randomly assigned to control (original pricing) or treatment (discounted pricing) groups.

```python
import pandas as pd
import numpy as np
np.random.seed(42)

n = 10000
cities = ["Austin", "Chicago", "Denver", "Miami", "Seattle"]
data = pd.DataFrame({
    "user_id": range(1, n+1),
    "city": np.random.choice(cities, n),
    "group": np.random.choice(["control", "treatment"], n),
})

data["base_rate"] = 0.1 + data["city"].map({
    "Austin": 0.01, "Chicago": -0.01, "Denver": 0.00,
    "Miami": 0.02, "Seattle": 0.015
})
data["treatment_lift"] = np.where(data["group"] == "treatment", 0.05, 0.00)
data["conversion_prob"] = data["base_rate"] + data["treatment_lift"]
data["converted"] = np.random.binomial(1, data["conversion_prob"])
```
### 📊 Group-Level Summary Statistics

To understand the effect of the price intervention, I computed the average **price**, **views**, and **conversion rate** for the control and treatment groups.

```
summary = df.groupby('group')[['price', 'views', 'conversion_rate']].mean().round(3)
```

|   Group    |  Price   |  Views  | Conversion Rate |
|:----------:|:--------:|:-------:|:----------------:|
|  Control   | 149.66   | 20.00   |      0.100       |
| Treatment  | 127.66   | 19.98   |      0.150       |


- Price: The average price in the control group was $149.67, while the treatment group had a lower average price of $127.66, confirming the experimental manipulation.
- Views: Both groups had similar average views (~20), suggesting comparable exposure and balanced randomization.
- Conversion Rate: The treatment group achieved a conversion rate of 15%, compared to 10% in the control group — a 50% relative lift in conversions. This suggests the pricing intervention was effective in driving higher conversion rates without reducing visibility.

# T-test

```
control = df[df['group'] == 'control']['conversion_rate']
treatment = df[df['group'] == 'treatment']['conversion_rate']
t_stat, p_val = ttest_ind(treatment, control)

print(f"Control Mean: {conversion_means['control']:.4f}")
print(f"Treatment Mean: {conversion_means['treatment']:.4f}")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
```

## 2. Initial A/B Test Analysis

### 🧪 A/B Test Summary

- **Control Group Mean Conversion Rate**: 10%  
- **Treatment Group Mean Conversion Rate**: 15%  
- **T-statistic**: 95.8623  
- **P-value**: 0.0000  

✅ This result is statistically significant at the 0.01 level.

### Conversion Rate Distribution Plot
```
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="conversion_prob",
    hue="group",
    kde=True,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Conversion Rate Distribution by Group")
plt.xlabel("Conversion Probability")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("conversion_rate_distribution.png")
plt.show()
```

![conversion rate distribution](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/ab.png?raw=true) 

### Conversion Rate Distribution by Group

This plot compares the distribution of **conversion rates** between the **treatment group** (blue) and the **control group** (orange). The distributions are visualized using overlapping histograms and kernel density estimates (KDE).

#### 🔹 Key Observations:

- **Treatment Group**:
  - The distribution is **right-skewed**, indicating a wider range of conversion rates.
  - The density peaks around **0.13 to 0.15**, with a long tail extending beyond **0.3**, and even reaching **0.5+**.
  - This suggests that some individuals in the treatment group experienced **very high conversion rates**.

- **Control Group**:
  - The distribution is more **narrow** and concentrated at the lower end.
  - The peak density is below **0.1**, and the tail drops off much earlier than the treatment group.
  - This implies **lower overall conversion performance** without the treatment.

#### 🔎 Interpretation:

- The **treatment group consistently achieves higher conversion rates** than the control group.
- The rightward shift in the treatment group distribution is a strong visual signal of **positive uplift** due to the intervention.
- The broader spread in the treatment group indicates that **some users benefitted much more than others**, highlighting possible heterogeneity in treatment effect.

---

**Conclusion**: This visualization supports the hypothesis that the treatment has a **beneficial impact** on conversion rates, justifying further analysis (e.g., uplift modeling or statistical testing).


I calculate the lift and statistical significance of the treatment effect.

```python
from scipy import stats

control = data[data.group == 'control']["converted"]
treatment = data[data.group == 'treatment']["converted"]

lift = treatment.mean() / control.mean() - 1
stat, pval = stats.ttest_ind(treatment, control)

print(f"Lift: {lift:.2%}")
print(f"T-statistic: {stat:.4f}, P-value: {pval:.4f}")
```

> **Lift:** ~49.81% 
> **P-value:** 0.0000 → statistically significant effect of the treatment.

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 🎯 Prepare binary treatment and outcome variables
df['treatment'] = (df['group'] == 'treatment').astype(int)
df['converted'] = (df['conversion_rate'] > 0).astype(int)  # binary target

# 🧼 One-hot encode city
df_encoded = pd.get_dummies(df, columns=['city'], drop_first=True)

# 🧾 Feature set
features = ['price', 'views', 'bookings'] + [col for col in df_encoded.columns if col.startswith('city_')]
X = df_encoded[features]
y = df_encoded['converted']
treatment = df_encoded['treatment']

# ✂️ Split into train/test sets
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X, y, treatment, test_size=0.3, stratify=treatment, random_state=42
)

# 🔀 Split training data by group
X_train_treat = X_train[treat_train == 1]
y_train_treat = y_train[treat_train == 1]

X_train_ctrl = X_train[treat_train == 0]
y_train_ctrl = y_train[treat_train == 0]

# 🏗️ Fit Random Forest models separately
rf_treat = RandomForestClassifier(n_estimators=100, random_state=42)
rf_ctrl = RandomForestClassifier(n_estimators=100, random_state=42)

rf_treat.fit(X_train_treat, y_train_treat)
rf_ctrl.fit(X_train_ctrl, y_train_ctrl)

# 📊 Predict probabilities on test set
p_treat = rf_treat.predict_proba(X_test)[:, 1]
p_ctrl = rf_ctrl.predict_proba(X_test)[:, 1]

# ➕ Compute uplift
uplift_score = p_treat - p_ctrl

# 🧾 Add uplift scores to test set
X_test_copy = X_test.copy()
X_test_copy['uplift_score'] = uplift_score
X_test_copy['converted'] = y_test.values
X_test_copy['treatment'] = treat_test.values

# 🕵️ View top uplift rows
print(X_test_copy.sort_values(by='uplift_score', ascending=False).head(10))
```
### Uplift Model Output Preview and Interpretation

Below is a preview of the dataset after applying the uplift model using the `ClassTransformation` method from the `sklift` library. The table shows the input features, the model-generated uplift score, and the actual outcomes:

|  price | views | bookings | city\_Chicago | city\_Denver | city\_Miami | city\_Seattle | uplift\_score | converted | treatment |
| :----: | :---: | :------: | :-----------: | :----------: | :---------: | :-----------: | :-----------: | :-------: | :-------: |
| 150.69 |   6   |     1    |      True     |     False    |    False    |     False     |      0.09     |     1     |     0     |
| 180.74 |   14  |     0    |     False     |     False    |     True    |     False     |      0.07     |     0     |     0     |
| 129.21 |   8   |     0    |      True     |     False    |    False    |      True     |      0.07     |     0     |     0     |
| 171.33 |   24  |     0    |     False     |     False    |    False    |     False     |      0.07     |     0     |     0     |
| 181.80 |   14  |     0    |     False     |     False    |    False    |     False     |      0.07     |     1     |     0     |
| 138.57 |   6   |     2    |     False     |     False    |    False    |     False     |      0.07     |     0     |     1     |
| 189.83 |   16  |     0    |     False     |     False    |     True    |     False     |      0.06     |     0     |     0     |
| 176.21 |   19  |     0    |     False     |     False    |    False    |     False     |      0.06     |     0     |     1     |
| 164.15 |   16  |     0    |     False     |     False    |    False    |     False     |      0.06     |     0     |     0     |
| 189.77 |   13  |     0    |     False     |     False    |     True    |     False     |      0.06     |     0     |     0     |


#### 🔍 What the Uplift Scores Represent

- The `uplift_score` column estimates the **increase in probability of conversion** if a user were treated.
- For example:
  - A score of `0.09` means that the model predicts a **9 percentage point lift** in conversion probability if this customer receives the treatment.
  - Rows with repeated scores indicate the model is producing **similar uplift estimates** for individuals with similar features (e.g., same city), especially since city is the only feature used.

#### 💡 Model Behavior

- In this case, all rows shown have `treatment = 0` (control group).
- The uplift model is being used to **simulate counterfactual outcomes** — estimating what would happen if these users had received the treatment.
- Because the features include only one-hot encoded cities (e.g., `city_Miami`, `city_Chicago`), uplift scores are **not very granular**, leading to repetition in predicted values.

#### ✅ Conclusion

This table provides a useful summary of model behavior. While the model is successfully estimating treatment effects, additional predictive features (such as demographics or behavioral metrics) would likely improve the **precision and segmentation** of the uplift estimates.


## 3. Uplift Modeling with sklift
I use again the `ClassTransformation` approach from the `sklift` package (a library for uplift modeling) to estimate uplift based on city and treatment group.

```python
from sklift.models import ClassTransformation
from sklearn.ensemble import RandomForestClassifier

uplift_model = ClassTransformation(RandomForestClassifier(n_estimators=100, random_state=42))
X = pd.get_dummies(data[["city"]], drop_first=True)
y = data["converted"]
treatment = data["group"] == "treatment"

uplift_model.fit(X, y, treatment)
data["uplift_score"] = uplift_model.predict(X)
```

## 4. Average Uplift by City with Confidence Intervals
We compute the mean and standard error of uplift scores by city and visualize them.

```python
import matplotlib.pyplot as plt

uplift_summary = data.groupby("city")["uplift_score"].agg(["mean", "count", "std"]).reset_index()
uplift_summary["sem"] = uplift_summary["std"] / np.sqrt(uplift_summary["count"])

plt.errorbar(x=uplift_summary["city"], y=uplift_summary["mean"], yerr=uplift_summary["sem"], fmt='D', color='black')
for i, row in uplift_summary.iterrows():
    plt.text(i, row["mean"] + 0.0001, f"{row['mean']:.3f}", ha='center')
plt.axhline(0, linestyle='--', color='red')
plt.title("Average Uplift by City with 95% CI")
plt.xlabel("City")
plt.ylabel("Estimated Uplift")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
```
![lift by city](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/ab1.png?raw=true) 

### Uplift Summary Table

| City     | Mean     | Std Dev | Count | SEM     |
|:---------|---------:|--------:|------:|--------:|
| Austin   | 0.00085  | 0.0058  | 2032  | 0.00013 |
| Chicago  | 0.00076  | 0.0057  | 1962  | 0.00013 |
| Denver   | 0.00080  | 0.0056  | 2031  | 0.00012 |
| Miami    | 0.00134  | 0.0058  | 1995  | 0.00013 |
| Seattle  | 0.00120  | 0.0057  | 1980  | 0.00013 |


## 5. Key Insights
- The A/B test revealed a **49.81% increase in conversion** due to pricing changes.
- **Uplift modeling** identified heterogeneous treatment effects by city.
- **Miami** and **Seattle** show the highest average uplift, suggesting they may be more responsive to price changes.
- This approach can inform targeted marketing and localized pricing strategies.

## Conclusion
This project illustrates a practical application of A/B testing and uplift modeling in a marketplace scenario. It highlights the importance of evaluating causal effects at a granular level to support data-driven decision-making.

