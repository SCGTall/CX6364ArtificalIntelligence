# -*- coding: utf-8 -*-
"""
Question 3:
Use the python library (sklearn.linear model) to train a logistic regression model for the Titanic dataset:
https://blog.goodaudience.com/machine-learning-using-logistic-regression-in-python-with-code-ab3c7f5f3bed.
Split the dataset to a training set (80% samples) and a testing set (20% samples). Report the overall classification
accuracies on the training and testing sets and report the precision, recall, and F-measure scores for each of the two
classes on the training and testing sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 1
DIAGRAM_DIR = "diagrams/q3"
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

print(df.shape)
print(df.dtypes)
print(df.sample(10))
# Since our homework required us to split the data into training set and testing set
columns = df.columns.tolist()
print("\nWe have the following columns:")
print(columns)

# original data
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
diagram_func("orignial.png")

# preprocess data
dropped_columns = []  # mark the dropped columns
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
# PassengerId
dropped_columns.append('PassengerId')
print("Drop 'PassengerId'.")  # PassengerId is just an id for a passenger, just drop it.
# Survived. Survived is the target will we learn. Keep it now but it will not be included in features.
# Pclass: Ticket class
print("'Pclass':")
print(df['Pclass'].value_counts())
print("Has NaN?: " + str(df['Pclass'].isnull().any()))
# Name
print("'Name':")
dropped_columns.append('Name')
print("Drop 'Name'.")  # Name has not relation with Survived. Drop it.
# Sex
print("'Sex':")
print(df['Sex'].value_counts())
print("Has NaN?: " + str(df['Sex'].isnull().any()))
df['Sex'].replace('male', 0, inplace=True)
df['Sex'].replace('female', 1, inplace=True)
print(df['Sex'].value_counts())
print("Map male and female to 0 and 1 in 'Sex'.")
# Age: Age in years
print("'Age':")
print(df['Age'].dtypes)
print("Has NaN?: " + str(df['Age'].isnull().any()))
sns.boxplot(x='Pclass', y='Age', data=df, palette='winter')
diagram_func("boxplot_age.png")
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
# SibSp: # of siblings / spouses aboard the Titanic
print("'SibSp':")
print(df['SibSp'].value_counts())
print("Has NaN?: " + str(df['SibSp'].isnull().any()))
# Parch: # of parents/ children aboard the Titanic
print("'Parch':")
print(df['Parch'].value_counts())
print("Has NaN?: " + str(df['Parch'].isnull().any()))
# Ticket: Ticket number
print("'Ticket':")
dropped_columns.append('Ticket')
print("Drop 'Ticket'.")  # Ticket has not relation with Survived. Drop it.
# Fare: Passenger fare
print("'Fare':")
print(df['Fare'].dtypes)
print("Has NaN?: " + str(df['Fare'].isnull().any()))
# Cabin: Cabin number
print("'Cabin':")
print("NaN percentage in Cabin: " + str(df['Cabin'].isna().sum()/df['Cabin'].shape[0]))
dropped_columns.append('Cabin')
print("Drop 'Cabin'.")  # Cabin has too many NaN values. Drop it
# Embarked: Port of Embarkation. C = Cherbourg, Q = Queenstown, S = Southampton
print("'Embarked':")
print(df['Embarked'].value_counts())
print("Has NaN?: " + str(df['Embarked'].isnull().any()))
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
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
diagram_func("processed.png")

y = df['Survived']
X = df.drop('Survived', axis=1)
features = X.columns.tolist()
print("feature_names:\n" + str(features))

# analysis of whole data
if SHOW_DIAGRAM or SAVE_DIAGRAM:
    diagram_y = y
    for i in range(len(features)):
        diagram_x = X[features[i]]
        plt.plot(diagram_x, diagram_y, '.',)
        plt.xlabel(features[i])
        plt.ylabel("Survived")
        diagram_func(str(i) + "_" + str(features[i]) + ".png")


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

# logistic regression and calculate precision_recall_fscore_support
logistic_regression = sklm.LogisticRegression(max_iter=200)
logistic_regression.fit(X_trn, y_trn)
print("Accuracies of training set: " + str(logistic_regression.score(X_trn, y_trn)))
print("Accuracies of testing set: " + str(logistic_regression.score(X_tst, y_tst)))
pred_trn = logistic_regression.predict(X_trn)
pred_tst = logistic_regression.predict(X_tst)
print("The precision, recall rates, and the F1-score of training set:")
print(precision_recall_fscore_support(y_trn, pred_trn))
print("The precision, recall rates, and the F1-score of testing set:")
print(precision_recall_fscore_support(y_tst, pred_tst))
# For survival, 0 = No, 1 = Yes.
