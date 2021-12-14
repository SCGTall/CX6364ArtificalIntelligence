# -*- coding: utf-8 -*-
"""
Question 1:
Use the python library (sklearn.linear.model) to train a linear regression model for the Boston housing dataset:
https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155.
Split the dataset to a training set (70% samples) and a testing set (30% samples). Report the root mean squared errors
(RMSE) on the training and testing sets.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt

SEED = 1
DIAGRAM_DIR = "diagrams/q1"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True


def diagram_func(filename):
    if not os.path.exists(DIAGRAM_DIR):
        os.makedirs(DIAGRAM_DIR)
    if SHOW_DIAGRAM:
        plt.show()
    if SAVE_DIAGRAM:
        plt.savefig(DIAGRAM_DIR + "/" + filename, bbox_inches='tight')
        plt.close()


# The Boston housing prices dataset has an ethical problem.
# The scikit-learn maintainers therefore strongly discourage the use of this dataset unless the purpose of the code is
# to study and educate about ethical issues in data science and machine learning. In studying case, you can fetch the
# dataset from the original source:

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep='\s+', skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq. ft",
    "INDUS": "Proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
    "NOX": "Nitric oxide concentration (parts per 10 million)",
    "RM": "Average number of rooms per dwelling",
    "AGE": "Proportion of owner-occupied units built prior to 1940",
    "DIS": "Weighted distances to five Boston employment centers",
    "RAD": "Index of accessibility to radial highways",
    "TAX": "Full-value property tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B": "1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town",
    "LSTAT": "Percentage of lower status of the population",
}

# The last row in feature_names which is "MEDV" is price which refers to y value
# "MEDV": "Median value of owner-occupied homes in $1000s"

features = [*feature_names]  # save all feature names in list. This line can save dic.keys() as list
print("data: contains the information for various houses\ntarget: prices of the house")
print("feature_names:\n" + str(features))

# analysis of whole data
print("\n'data':")
print(data.shape)
print(str(data[:3]) + "\n...")
print("'target':")
print(target.shape)
print(str(target[:3]) + "\n...\n")
print("Check if data needs imputation:")
print("'data': " + str(True in np.isnan(data)))
print("'target': " + str(True in np.isnan(target)))

if SHOW_DIAGRAM or SAVE_DIAGRAM:
    diagram_y = target
    for i in range(len(features)):
        diagram_x = data[:, i]
        plt.plot(diagram_x, diagram_y, '.')
        plt.xlabel(features[i])
        plt.ylabel("MEDV")
        diagram_func(str(i) + "_" + str(features[i]) + ".png")


# split training set and testing set
ones = np.ones(len(target))
X = np.c_[ones, data]
y = target
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3, random_state=SEED)
print("Training set size:" + str(len(X_trn)))
print("Testing set size:" + str(len(X_tst)))

# inspection of training data
print("X: " + str([1] + features))
print(X_trn.shape)
print(str(X_trn[:3]) + "\n...\n")
print("y: ['MEDV']")
print(y_trn.shape)
print(str(y_trn[:3]) + "\n...\n")

# linear regression and calculate RMSE
linear_regression = sklm.LinearRegression()
linear_regression.fit(X_trn, y_trn)
pred_trn = linear_regression.predict(X_trn)
pred_tst = linear_regression.predict(X_tst)
print("RMSE of training set: " + str(mean_squared_error(y_trn, pred_trn) ** 0.5))
print("RMSE of testing set: " + str(mean_squared_error(y_tst, pred_tst) ** 0.5))
