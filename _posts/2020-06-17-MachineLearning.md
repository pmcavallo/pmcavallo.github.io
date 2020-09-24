---
layout: post
title: Machine Learning
---

This is an example of a machine learning model in both **R** and **Python**:

More specifically, we're going to be developing a *Random Forest* model, which is an *Ensemble Learning* machine learning method. Ensemble learning is when you take multiple machine learning models to create one larger machine learning algorithm. The random forest method combines many *decision trees* models together. 

The example we're going to be using is, as always, with foreign direct investment (FDI) data. The data is at the metropolitan area-level for the 50 largest metro areas in the US. The dependent variable is binary with 1 if the metro area received an FDI project in headquarters (HQ) that year and zero otherwise. Headquarters projects are the most sought after by city officials, and by splitting the data and using only HQ FDI we also get the a lot more zeros in the data.

The features (independent variables) to be considered are **GDP per capita**, measured as the per capita real GDP in chained 2009 dollars, **housing prices**, measured as the median home value per square foot, and **education**, measured as the percentage of the population with a bachelor's degree or higher. For simplicity sake, we're only going to use 3 features, but we know that an FDI model would require a higher number of predictors. We're using 3 variables that have proven to be significant predictors of FDI in the past, but, for now, we are also leaving many other variables usually required in an FDI model. We have 658 observations for 50 metro areas over 14 years. The goal is to predict if metro areas with a certain combination of the above-mentioned features will receive FDI in HQs.

We'll start with Pyhton. First we bring in the data, identify the dependent and independent variables and split it into training and test set:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('/Users/Paola/Desktop/Paulo/github.csv')
IV = data.iloc[:, 3:-1].values
DV = data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
IV_train, IV_test, DV_train, DV_test = train_test_split(IV, DV, test_size = 0.20, random_state = 1)

```

We choose to split the dataset into 80 percent for training and 20 for test, and we add the *random state* to make the results reproducible. Next we proceed to train the *Random Forest Classification* model on the training set, as follows:

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state = 1)
rf.fit(IV_train, DV_train)
```
We first change the number of trees (*n_estimators*) to 1,000 from the default number of 100 trees. We use *entropy* as the criterion for the information gain and *random state* again to make it reproducible. We then quickly check what would the model predict in some situations. We first check the descriptive statistics of the features:

![Network Plot](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/features.png?raw=true)

The first situation would look at a metro area with U$ 80 dolars as the square foot price of homes, and we see that U$ 131 is the mean, so we're way below the average implying a metro area with a cheap housing market. We set the GDP per capita to U$ 60,000, which is slightly above the mean of U$ 53,962. Lastly, we set the education level to 0.25, above the mean of 0.21. The second situation sees a metro area with the same GDP per capita and housing prices, but lower the education lervel to 0.11, closer to the minimun of 0.10:


```python
print(rf.predict([[80, 60000, 0.25]]))
[1]

print(rf.predict([[80, 60000, 0.11]]))
[0]
```

The first situation predicts the metro area would receive the HQ investment while the second situation predicts the metro area would not. So then we visually compare some of the predicted results to the actual results in the test set to quicly see how the model did:

```python
y_pred = rf.predict(IV_test)
```

![Predicted Results](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/concatenate.png?raw=true)

The model seemed to have done well, we see only two instances where the prediction didn't match the test set, it's worth reminding that these are only the top results, there are many more observations. Let's have a look at the confusion matrix and the accuracy of the model:

```python
#Making the confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(DV_test, y_pred)
array([[33, 17],
       [19, 63]])
       
accuracy_score(DV_test, y_pred)
Out[156]: 0.7272727272727273
```
As expected, the model didn't do that well. The confusion matrix show many prediction mistakes both in zeros and ones and the model accuracy is very low at 0.72.  We simplified the model too much for this example, but it shows how to build a random forest classification model in a short and clear manner. The next step is to try many more features and apply a feature selection. We will also try different classification models, such as K-nearest neighbors and kernel SVM. Thanks!
