# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.utils.data as tud

SEED = 1
DIAGRAM_DIR = "diagrams/q1"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True

np.random.seed(SEED)
torch.manual_seed(SEED)


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
ones = np.ones(len(target))
X = np.c_[ones, data]
y = target
ds = DatasetPlus(X, y)
trn, tst = ds.split_data(test_ratio=0.3)
print(type(X), type(y))
print("X size: " + str(X.shape))
print("y size: " + str(y.shape))
print("Training set: " + str(len(trn)))
print("Testing set: " + str(len(tst)))
trn_dl = tud.DataLoader(trn, batch_size=32, shuffle=True)  # batch size = 32
tst_dl = tud.DataLoader(tst, batch_size=32)


class Model(nn.Module):
    def __init__(self, _fin):
        super(Model, self).__init__()
        # Two hidden layers: the first hidden layer must contain 16 units using ReLU activation function;
        # the second layer must contain 32 units using tanh activation function.
        _f1, _f2, _fout = 16, 32, 1
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
        preds.append(pred)
        actuals.append(actual)
    preds, actuals = np.vstack(preds), np.vstack(actuals)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return rmse


# Train model and test it
print("\nTraining...\n")
fin = X.shape[1]
print("Input feature: " + str(fin))
model = Model(_fin=fin)
train(trn_dl, model)
print("RMSE of training set:")
print(evaluate(trn_dl, model))
print("RMSE of testing set:")
print(evaluate(tst_dl, model))
