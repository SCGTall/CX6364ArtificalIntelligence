# -*- coding: utf-8 -*-
"""
Question 4:
Based on question 3, Implement the following five algorithms to train a logistic regression model for the Titantic
dataset. Report the overall classification accuracies on the training and testing sets and report the precision,
recall, and F-measure scores for each of the two classes on the training and testing sets.
1. The gradient descent algorithm
2. The stochastic gradient descent (SGD) algorithm
3. The SGD algorithm with momentum
4. The SGD algorithm with Nesterov momentum
5. The AdaGrad algorithm

Elminate some part for data analysis. Check in question3.py if you want.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

SEED = 1
DIAGRAM_DIR = "diagrams/q4"
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


url = "LogisticRegression-master/titanic_train.csv"

df = pd.read_csv(url)
# Since our homework required us to split the data into training set and testing set
columns = df.columns.tolist()

# preprocess data
dropped_columns = []  # mark the dropped columns
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
dropped_columns.append('PassengerId')  # PassengerId is just an id for a passenger, just drop it.
dropped_columns.append('Name')
df['Sex'].replace('male', 0, inplace=True)
df['Sex'].replace('female', 1, inplace=True)
print("Map male and female to 0 and 1 in 'Sex'.")
print(df[['Pclass', 'Age']].groupby('Pclass').mean())
#   Pclass    Age
#   1         38.23
#   2         29.88
#   3         25.14


def impute_age(cols):  # impute age by the average age of each pclass
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38.23
        elif pclass == 2:
            return 29.88
        elif pclass == 3:
            return 25.14
        else:
            return 0
    else:
        return age


df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)
dropped_columns.append('Ticket')
dropped_columns.append('Cabin')
df['Embarked'].replace('C', 0, inplace=True)
df['Embarked'].replace('Q', 1, inplace=True)
df['Embarked'].replace('S', 2, inplace=True)
df = df.fillna({'Embarked': 2})
df[['Embarked']] = df[['Embarked']].astype(int)
print("Map C, Q, S to 0, 1 and 2 in 'Embarked' and impute with mode = 2.")

for column in dropped_columns:
    df.drop(column, axis=1, inplace=True)
print(df.shape)
print(df.dtypes)
print(df.sample(10))

y = df['Survived']
X = df.drop('Survived', axis=1)
features = X.columns.tolist()
print("feature_names:\n" + str(features))

# split training set and testing set
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=SEED)
print("Training set size:" + str(len(X_trn)))
print("Testing set size:" + str(len(X_tst)))

# inspection of training data
print("X: " + str([1] + features))
print(X_trn.shape)
print(str(X_trn[:3]) + "\n...\n")
print("y: ['Survived']")
print(y_trn.shape)
print(str(y_trn[:3]) + "\n...\n")


def sigmoid(v):
    return 1 / ((1 + np.exp(-v.astype('float128'))).astype('float'))  # float128 is to prevent possible overflow here


def classify(v):
    if v <= 0.5:
        return 0
    else:
        return 1


algorithms = ["GD", "SGD", "SGDwithMomentum", "SGDwithNesterovMomentum", "AdaGrad"]


class MyLogisticRegression:
    def __init__(self, _a_id, _alpha=1e-2, _steps=1e3, _batch=50, _eta=0.9, _epsilon=1e-6):

        if a_id not in range(5):
            print("Unvalid algorithm id.")
            return
        np.random.seed(SEED)
        self.a_id = _a_id

        self.alpha = _alpha
        self.steps = int(_steps)

        # minibatch
        self.batch = _batch

        self.theta = None

        # momentum
        self.eta = _eta

        # AdaGrad
        self.epsilon = _epsilon

    def fit(self, _X_trn, _y_trn):
        # init
        N, p = _X_trn.shape[0], _X_trn.shape[1] + 1  # x0-xp for N data points
        theta = np.zeros([1, p])
        ones = np.ones([N, 1])
        X = np.concatenate((ones, _X_trn), axis=1)
        Y = np.asmatrix(_y_trn).transpose()
        v = np.zeros([1, p])
        r = np.zeros([1, p])
        for t in range(self.steps):
            x = X
            y = Y
            # if SGD, use minibatch
            if self.a_id != 0:
                top_id = np.random.randint(N)
                bottom_id = top_id + self.batch
                x = X[top_id: bottom_id, :]
                y = Y[top_id: bottom_id, :]
            momt = np.zeros([1, p])
            # if use momentum step
            if self.a_id in [2, 3]:
                momt = np.dot(self.eta, v)
            h = sigmoid(np.dot(theta, x.transpose()))
            # if Nesterov
            if self.a_id == 3:
                h = sigmoid(np.dot(np.add(theta, np.dot(self.eta, v)), x.transpose()))
            diff = np.subtract(h, y.transpose())
            g = np.dot(diff, x) / x.shape[0]
            grad = np.dot(self.alpha, g)
            # if AdaGrad
            if self.a_id == 4:
                r = np.add(r, np.multiply(g, g))
                a = self.alpha / (np.sqrt(self.epsilon + r))
                grad = np.multiply(a, g)
            if np.isnan(grad).any():  # jump out since theta is meaningless now
                theta = np.multiply(theta, np.nan)
                break
            v = np.subtract(momt, grad)
            if np.isnan(v).any():  # jump out since theta is meaningless now
                theta = np.multiply(theta, np.nan)
                break
            theta = np.add(theta, v)
        self.theta = theta  # save theta and quit
        return self

    @staticmethod
    def score(_data, _pred):
        count = 0
        length = min(len(_data), len(_pred))
        for i in range(length):
            if _data.iloc[i] == _pred[i][0]:
                count += 1
        return count / length

    def predict(self, _X_test):
        N2 = _X_test.shape[0]
        ones2 = np.ones([N2, 1])
        X = np.concatenate((ones2, _X_test), axis=1)
        y = sigmoid(np.dot(X, self.theta.transpose()))
        pred = []
        for i in range(len(y)):
            pred.append(classify(y[i][0]))
        return np.asmatrix(pred).transpose().A


def psfs_stable(_data, _pred):
    if np.isnan(_pred).any():  # bad features may lead to non-convergence
        return None
    else:
        return precision_recall_fscore_support(_data, _pred, average='weighted', warn_for=tuple())


# logistic regression and calculate precision_recall_fscore_support
for a_id in range(len(algorithms)):
    print("\nAlgorithm = " + algorithms[a_id] + ":")
    record = []
    prfs_record = []
    start = 0
    end = -8
    for exp in range(start, end, -1):
        alpha = 10 ** exp
        logistic_regression = MyLogisticRegression(a_id, _alpha=alpha)
        logistic_regression.fit(X_trn, y_trn)
        pred_trn = logistic_regression.predict(X_trn)
        pred_tst = logistic_regression.predict(X_tst)
        acc_trn = logistic_regression.score(y_trn, pred_trn)
        acc_tst = logistic_regression.score(y_tst, pred_tst)
        # For survival, 0 = No, 1 = Yes.
        if acc_trn and acc_tst:
            prfs_trn = psfs_stable(y_trn, pred_trn)
            prfs_tst = psfs_stable(y_tst, pred_tst)
            record.append([exp, acc_trn, acc_tst])
            prfs_record.append([prfs_trn, prfs_tst])
    if SHOW_DIAGRAM or SAVE_DIAGRAM:
        datapoints = np.asarray(record)
        diagram_x = datapoints.transpose()[0].tolist()
        for i in range(1, datapoints.shape[1]):
            diagram_y = datapoints.transpose()[i].tolist()
            plt.plot(diagram_x, diagram_y)
        plt.xlim(end, start + 1)
        plt.xlabel("alpha (10^x)")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(algorithms[a_id])
        plt.legend(["training set", "testing set"])
        diagram_func(str(a_id) + "_" + algorithms[a_id] + "_" + "learning_rates.png")
    # report best
    if len(record) == 0:
        print("No converged result found")
    else:
        mem = record[0]
        prfs_men = prfs_record[0]
        for i in range(1, len(record)):
            if sum(mem[1:]) < sum(record[i][1:]):
                mem = record[i]
                prfs_men = prfs_record[i]
        print("Temporary largest accuracies for " + algorithms[a_id] + ":")
        print("leaning rate (alpha): " + str(10 ** mem[0]))
        print("training set: " + str(mem[1]))
        print("The precision, recall rates, and the F1-score of training set:")
        print(prfs_men[0])
        print("testing set: " + str(mem[2]))
        print("The precision, recall rates, and the F1-score of training set:")
        print(prfs_men[1])
