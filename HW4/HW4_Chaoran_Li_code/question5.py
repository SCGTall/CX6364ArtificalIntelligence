# -*- coding: utf-8 -*-
"""
Question 5:
Implement the Adam algorithm to train a linear regression model for the Boston housing data set. Report the root mean
squared errors (RMSE) on the training and testing sets.

Use minibatch here.

Elminate some part for data analysis. Check in question1.py if you want.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

SEED = 1
DIAGRAM_DIR = "diagrams/q5"
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
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
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

algorithms = ["GD", "SGD", "SGDwithMomentum", "SGDwithNesterovMomentum", "AdaGrad", "Adam"]


class MyLinearRegression:
    def __init__(self, _a_id, _alpha=1e-3, _steps=1e3, _batch=50, _epsilon=1e-8, _rho1=0.9, _rho2=0.999):

        if a_id not in range(6):
            print("Unvalid algorithm id.")
            return
        np.random.seed(SEED)
        self.a_id = _a_id

        self.alpha = _alpha
        self.steps = int(_steps)

        # minibatch
        self.batch = _batch

        self.theta = None

        # Adam
        self.epsilon = _epsilon
        self.rho1 = _rho1
        self.rho2 = _rho2

    def fit(self, _X_trn, _y_trn):
        # init
        N, p = _X_trn.shape[0], _X_trn.shape[1] + 1  # x0-xp for N data points
        theta = np.zeros([1, p])
        ones2 = np.ones([N, 1])
        X = np.concatenate((ones2, _X_trn), axis=1)
        Y = np.asmatrix(_y_trn).transpose()
        v = np.zeros([1, p])
        r = np.zeros([1, p])
        s = np.zeros([1, p])
        for t in range(1, self.steps + 1):  # t = 0 will cause divide by zero in bias correction step
            top_id = np.random.randint(N)
            bottom_id = top_id + self.batch
            x = X[top_id: bottom_id, :]
            y = Y[top_id: bottom_id, :]
            h = np.dot(theta, x.transpose())
            diff = np.subtract(h, y.transpose())
            g = np.dot(diff, x) / x.shape[0]
            s = np.add(np.dot(self.rho1, s), np.dot(1 - self.rho1, g))
            r = np.add(np.dot(self.rho2, r), np.dot(1 - self.rho2, np.multiply(g, g)))
            s_hat = s / (1 - self.rho1 ** t)
            r_hat = r / (1 - self.rho2 ** t)
            a = self.alpha / np.sqrt(np.add(r_hat, self.epsilon))
            v = -np.multiply(a, s_hat)
            if np.isnan(v).any():  # jump out since theta is meaningless now
                theta = np.multiply(theta, np.nan)
                break
            theta = np.add(theta, v)
        self.theta = theta  # save theta and quit
        return self

    def predict(self, _X_test):
        N2 = _X_test.shape[0]
        ones3 = np.ones([N2, 1])
        X = np.concatenate((ones3, _X_test), axis=1)
        return np.dot(X, self.theta.transpose()).A


def rmse_stable(_data, _pred):
    if np.isnan(_pred).any():  # bad features may lead to non-convergence
        return None
    else:
        return mean_squared_error(_data, _pred) ** 0.5


# linear regression and calculate RMSE
a_id = 5
print("\nAlgorithm = " + algorithms[a_id] + ":")
record = []
start = -1
end = -12
for exp in range(start, end, -1):
    alpha = 10 ** exp
    linear_regression = MyLinearRegression(a_id, _alpha=alpha)
    linear_regression.fit(X_trn, y_trn)
    pred_trn = linear_regression.predict(X_trn)
    pred_tst = linear_regression.predict(X_tst)
    rmse_trn = rmse_stable(y_trn, pred_trn)
    rmse_tst = rmse_stable(y_tst, pred_tst)
    if rmse_trn and rmse_tst:
        record.append([exp, rmse_trn, rmse_tst])
# diagram alpha v.s. RMSE
if SHOW_DIAGRAM or SAVE_DIAGRAM:
    datapoints = np.asarray(record)
    diagram_x = datapoints.transpose()[0].tolist()
    for i in range(1, datapoints.shape[1]):
        diagram_y = datapoints.transpose()[i].tolist()
        plt.plot(diagram_x, diagram_y)
    plt.xlim(end, start+1)
    plt.xlabel("alpha (10^x)")
    plt.ylim(0, 30)
    plt.ylabel("RMSE")
    plt.title(algorithms[a_id])
    plt.legend(["training set", "testing set"])
    diagram_func(str(a_id) + "_" + algorithms[a_id] + "_" + "learning_rates.png")
# report best RMSE
if len(record) == 0:
    print("No converged result found")
else:
    mem = record[0]
    for i in range(1, len(record)):
        if sum(mem[1:]) > sum(record[i][1:]):
            mem = record[i]
    print("Temporary best RMSE for " + algorithms[a_id] + ":")
    print("leaning rate (alpha): " + str(10 ** mem[0]))
    print("training set: " + str(mem[1]))
    print("testing set: " + str(mem[2]))
