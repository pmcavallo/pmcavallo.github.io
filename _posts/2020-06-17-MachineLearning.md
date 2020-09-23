---
layout: post
title: Machine Learning
---

This is an example of a machine learning model in both **R** and **Python**:

More specifically, we're going to be developing a *Random Forest* model, which is an *Ensemble Learning* machine learning method. Ensemble learning is when you take multiple machine learning models to create one larger machine learning algorithm. The random forest method combines many *decision trees* models together. 

The example we're going to be using is, as always, with foreign direct investment (FDI) data. The data is at the metropolitan area-level for the 50 largest metro areas in the US. The dependent variable is binary with 1 if the metro area received an FDI project in headquarters (HQ) that year and zero otherwise. Headquarters projects are the most sought after by city officials, and by splitting the data and using only HQ FDI we also get the a lot more zeros in the data.

The features (independent variables) to be considered are **GDP per capita**, measured as the per capita real GDP in chained 2009 dollars, **housing prices**, measured as the median home value per square foot, and **education**, measured as the percentage of the population with a bachelor's degree or higher. We have 658 observations for 50 metro areas over 14 years. The goal is to predict if metro areas with a certain combination of the above-mentioned features will receive FDI in HQs.

We'll start with Pyhton. First we bring in the data, identify the dependent and independent variables and split it into training and test set:

```Python
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

We choose to split the dataset intp 80 percent for training and 20 for test, and we add the *random state* to make the results reproducible.
