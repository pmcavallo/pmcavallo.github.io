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

| group     | price   | views  | conversion\_rate |
| --------- | ------- | ------ | ---------------- |
| control   | 149.666 | 20.003 | 0.100            |
| treatment | 127.661 | 19.986 | 0.150            |

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
print(f"Lift: {(conversion_means['treatment'] - conversion_means['control']) / conversion_means['control']:.2%}")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
```

## 2. Initial A/B Test Analysis

### 🧪 A/B Test Summary

- **Control Group Mean Conversion Rate**: 11.32%  
- **Treatment Group Mean Conversion Rate**: 16.33%  
- **Lift**: 44.23%  
- **T-statistic**: 7.2673  
- **P-value**: 3.942e-13  

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

## 3. Uplift Modeling with scikit-uplift
I use the `ClassTransformation` approach from the `sklift` packagee (a library for uplift modeling) to estimate uplift based on city and treatment group.

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

