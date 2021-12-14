# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.utils.data as tud

SEED = 1
DIAGRAM_DIR = "diagrams/q2"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True

np.random.seed(SEED)
torch.manual_seed(SEED)


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


# prepare data for Pytorch
class DatasetPlus(tud.Dataset):
    def __init__(self, _X, _y):
        self.X = _X.astype(np.float32)
        self.y = _y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def split_data(self, test_ratio):
        test_size = round(test_ratio * len(self.X))
        train_size = len(self.X) - test_size
        return tud.random_split(self, [train_size, test_size])


print("\nPrepare data for Pytorch")
y = df['Survived'].to_numpy()
fd_x = df.drop('Survived', axis=1)
features = fd_x.columns.tolist()
print("feature_names:\n" + str(features))
X = fd_x.to_numpy()
print(type(X), type(y))
ds = DatasetPlus(X, y)
trn, tst = ds.split_data(test_ratio=0.2)
print("X size: " + str(X.shape))
print("y size: " + str(y.shape))
print("Training set: " + str(len(trn)))
print("Testing set: " + str(len(tst)))
trn_dl = tud.DataLoader(trn, batch_size=32, shuffle=True)  # batch size = 32
tst_dl = tud.DataLoader(tst, batch_size=32)


class Model(nn.Module):
    def __init__(self, _fin):
        super(Model, self).__init__()
        # Two hidden layers: the first hidden layer must contain 5 units using ReLU activation function;
        # the second layer must contain 3 units using tanh activation function.
        _f1, _f2, _fout = 5, 3, 1
        self.l1 = nn.Linear(_fin, _f1)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(_f1, _f2)
        self.act2 = nn.Tanh()
        self.out = nn.Linear(_f2, _fout)

    def forward(self, X):
        X = self.l1(X)
        X = self.act1(X)
        X = self.l2(X)
        X = self.act2(X)
        X = self.out(X)
        return X


def train(_dl, _model, _alpha=1e-3, _eta=0.9, _epochs=10):
    # alpha: learning rate
    # eta: momentum
    # defaut: mean
    loss_func = nn.MSELoss()
    # SGD w/ momentum
    optimizer = torch.optim.SGD(_model.parameters(), lr=_alpha, momentum=_eta)

    for epoch in range(_epochs):
        for (i, (inputs, targets)) in enumerate(_dl):
            targets = targets.unsqueeze(1)  # different size
            optimizer.zero_grad()
            pred = _model(inputs)
            loss = loss_func(pred, targets)
            loss.backward()
            optimizer.step()


def evaluate(_dl, _model):
    preds, actuals = [], []
    for (i, (inputs, targets)) in enumerate(_dl):
        pred = _model(inputs).detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))

        # round up into binary result
        pred = pred.round()

        preds.append(pred)
        actuals.append(actual)
    preds, actuals = np.vstack(preds), np.vstack(actuals)
    acc = precision_recall_fscore_support(actuals, preds, average='weighted', warn_for=tuple())
    return acc


# Train model and test it
print("\nTraining...\n")
fin = X.shape[1]
print("Input feature: " + str(fin))
model = Model(_fin=fin)
train(trn_dl, model)
print("The precision, recall rates, and the F1-score of training set:")
print(evaluate(trn_dl, model))
print("The precision, recall rates, and the F1-score of testing set:")
print(evaluate(tst_dl, model))
