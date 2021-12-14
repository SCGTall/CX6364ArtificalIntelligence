# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
import cv2
import matplotlib.pyplot as plt
import torchvision as tv
from torchvision.datasets import FashionMNIST as minst
from sklearn.metrics import accuracy_score

SEED = 1
DIAGRAM_DIR = "diagrams/q4"
if not os.path.exists(DIAGRAM_DIR):
    os.makedirs(DIAGRAM_DIR)
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


np.random.seed(SEED)
torch.manual_seed(SEED)
epochs = 25
batch_size = 128
data_folder = "./data"
# transforms convert image from [0, 255] to [0, 1]
trn_ds = minst(root=data_folder, train=True, download=True, transform=tv.transforms.ToTensor())
tst_ds = minst(root=data_folder, train=False, download=True, transform=tv.transforms.ToTensor())
train_set_size = int(0.9 * len(trn_ds))
trn_ds, val_ds = tud.random_split(trn_ds, [train_set_size, len(trn_ds) - train_set_size])
print("Training set: " + str(len(trn_ds)))
print("Validation set: " + str(len(val_ds)))
print("Testing set: " + str(len(tst_ds)))
trn_dl = tud.DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
val_dl = tud.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
tst_dl = tud.DataLoader(tst_ds, batch_size=batch_size, shuffle=False)


def sampling(_labels, _ds):
    samples_ = {}
    index = 0
    while len(samples_) < len(_labels):
        (img, y) = _ds[index]
        label = _labels[y]
        if label not in samples_.keys():
            samples_[label] = (img, y)
        index += 1
    return samples_


def visualize_samples(_labels, _samples, _rows, _columns):
    fig, ax = plt.subplots(_rows, _columns, figsize=(13, 6))
    for r in range(_rows):
        for c in range(_columns):
            label = _labels[r * _columns + c]
            (img, y) = _samples[label]
            ax[r, c].imshow(cv2.cvtColor(np.transpose(img.numpy()), cv2.COLOR_GRAY2RGB))
            ax[r, c].set_title(label)
    diagram_func("samples.png")


# sample from data
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
samples = sampling(labels, trn_ds)
visualize_samples(labels, samples, 2, 5)


# train models
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input channel, output channel, kernel
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc = nn.Linear(in_features=196, out_features=10, bias=True)

    def forward(self, X):
        # n x 1 x 28 x 28
        X = self.conv1(X)
        # n x 4 x 28 x 28
        X = self.bn1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # n x 4 x 14 x 14
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # n x 4 x 7 x 7
        X = X.reshape(-1, 196)
        # n x 196
        X = self.fc(X)
        # n x 10
        return X


def train(_trn, _val, _model, _epochs, _alpha=1e-3, _eta=0.9):
    print("\nTraining:")
    log_ = {
        "Epoch": [],
        "Training Loss": [],
        "Training Accuracy": [],
        "Validation Loss": [],
        "Validation Accuracy": [],
    }

    for epoch in range(1, _epochs + 1):
        print("Epoch " + str(epoch) + ":")
        log_["Epoch"].append(epoch)

        for (i, (input, target)) in enumerate(_trn):
            optimizer.zero_grad()
            pred = _model(input)
            loss = loss_func(pred, target)
            loss.backward()
            optimizer.step()

        # record loss and accuracy in log
        acc_trn, loss_trn = evaluate(_trn, _model)
        log_["Training Loss"].append(loss_trn)
        log_["Training Accuracy"].append(acc_trn)
        acc_val, loss_val = evaluate(_val, _model)
        log_["Validation Loss"].append(loss_val)
        log_["Validation Accuracy"].append(acc_val)
        print("Training Loss:" + str(loss_trn))
        print("Validation Loss:" + str(loss_val))
        print("Training Accuracy:" + str(acc_trn))
        print("Validation Accuracy:" + str(acc_val))
    return log_


def evaluate(_dl, _model):
    preds, actuals = [], []
    outputs, targets = None, None
    for (i, (input, target)) in enumerate(_dl):
        if targets is not None:
            targets = torch.cat((targets, target), 0)
        else:
            targets = target
        output = _model(input)
        if outputs is not None:
            outputs = torch.cat((outputs, output), 0)
        else:
            outputs = output
        _, pred = torch.max(output.data, 1)
        pred = pred.tolist()
        actual = target.numpy()
        preds.extend(pred)
        actuals.extend(actual)
    loss = loss_func(outputs, targets).item()
    acc = accuracy_score(actuals, preds)
    return acc, loss


def visualize_log(_log):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(log["Epoch"], log["Training Loss"], '-r', label="Training")
    ax[0].plot(log["Epoch"], log["Validation Loss"], '-b', label="Validation")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Cross Entropy Loss")
    ax[0].legend()
    ax[1].plot(log["Epoch"], log["Training Accuracy"], '-r', label="Training")
    ax[1].plot(log["Epoch"], log["Validation Accuracy"], '-b', label="Validation")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim([0.6, 1])
    ax[1].legend()
    diagram_func("result.png")


model = Model()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
log = train(trn_dl, val_dl, model, epochs)
visualize_log(log)
acc_tst, loss_tst = evaluate(tst_dl, model)
print("\nTesting:")
print("Testing Loss:" + str(loss_tst))
print("Testing Accuracy:" + str(acc_tst))
