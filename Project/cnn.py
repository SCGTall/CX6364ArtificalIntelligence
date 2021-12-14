# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as tud
from sklearn.model_selection import train_test_split

DATA_DIR = "data/CTImagesInCOVID19_p2/"
DIAGRAM_DIR = "diagrams/" + os.path.basename(__file__).replace(".py", "")
OUTPUT_DIR = "data/Processed/"
IMAGES_DIR = OUTPUT_DIR + "images_short.npy"
RECONSTRUCTIONS_DIR = OUTPUT_DIR + "reconstructions_short.npy"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True
SEED = 42


def current_timestamp():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def check_folder(_folder):
    if not os.path.exists(_folder):
        os.makedirs(_folder)


def diagram_func(_filename):
    check_folder(DIAGRAM_DIR)
    if SHOW_DIAGRAM:
        plt.show()
    if SAVE_DIAGRAM:
        plt.savefig(DIAGRAM_DIR + "/" + _filename, bbox_inches='tight')
        plt.close()


def brief_check_image(_img):
    print(_img[::64, ::64])


def load_dataset(_dir):
    arr = np.load(_dir)
    print(arr.shape)
    return arr.tolist()


np.random.seed(SEED)
torch.manual_seed(SEED)
epochs = 60
batch_size = 8
# transforms convert image from [0, 255] to [0, 1]
current_timestamp()
print("Load data:")
imgs = load_dataset(IMAGES_DIR)
recons = load_dataset(RECONSTRUCTIONS_DIR)
# training: validation: testing = 8: 1: 1
print("Split data:\ntraining: validation: testing = 8: 1: 1")
X_trn, X_tst, y_trn, y_tst = train_test_split(recons, imgs, test_size=0.2, random_state=SEED)
X_tst, X_val, y_tst, y_val = train_test_split(X_tst, y_tst, test_size=0.5, random_state=SEED)
trn_ds = tud.TensorDataset(torch.Tensor(X_trn).unsqueeze(1), torch.Tensor(y_trn).unsqueeze(1))
val_ds = tud.TensorDataset(torch.Tensor(X_val).unsqueeze(1), torch.Tensor(y_val).unsqueeze(1))
tst_ds = tud.TensorDataset(torch.Tensor(X_tst).unsqueeze(1), torch.Tensor(y_tst).unsqueeze(1))
print("Training set: " + str(len(trn_ds)))
print("Validation set: " + str(len(val_ds)))
print("Testing set: " + str(len(tst_ds)))
trn_dl = tud.DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
val_dl = tud.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
tst_dl = tud.DataLoader(tst_ds, batch_size=batch_size, shuffle=False)


# train models
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input channel, output channel, kernel
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.bn1 = nn.BatchNorm2d(4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 6, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.bn2 = nn.BatchNorm2d(6, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act2 = nn.ReLU()
        self.conv_tr3 = nn.ConvTranspose2d(6, 4, kernel_size=(3, 3))
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act3 = nn.ReLU()
        self.conv_tr4 = nn.ConvTranspose2d(4, 1, kernel_size=(3, 3))
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.bn4 = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act4 = nn.ReLU()

    def forward(self, X):
        X = self.conv1(X)
        X = self.pool1(X)
        X = self.bn1(X)
        X = self.act1(X)
        X = self.conv2(X)
        X = self.pool2(X)
        X = self.bn2(X)
        X = self.act2(X)
        X = self.conv_tr3(X)
        X = self.unpool3(X)
        X = self.bn3(X)
        X = self.act3(X)
        X = self.conv_tr4(X)
        X = self.unpool4(X)
        X = self.bn4(X)
        X = self.act4(X)
        return X


def train(_trn, _val, _model, _epochs):
    print("\nTraining:")
    log_ = {
        "Epoch": [],
        "Training Loss": [],
        "Training Accuracy": [],
        "Validation Loss": [],
        "Validation Accuracy": [],
    }

    for epoch in range(1, _epochs + 1):
        print("\nEpoch " + str(epoch) + ":")
        current_timestamp()
        log_["Epoch"].append(epoch)

        for (i, (input, target)) in enumerate(_trn):
            optimizer.zero_grad()
            pred = _model(input)
            loss = loss_func(pred, target)
            loss.backward()
            optimizer.step()
        # record loss and accuracy in log
        print("Evaluate training set")
        current_timestamp()
        acc_trn, loss_trn = evaluate(_trn, _model)
        log_["Training Loss"].append(loss_trn)
        log_["Training Accuracy"].append(acc_trn)
        print("Training Loss:" + str(loss_trn))
        print("Training Accuracy:" + str(acc_trn))
        print("Evaluate validation set")
        current_timestamp()
        acc_val, loss_val = evaluate(_val, _model)
        log_["Validation Loss"].append(loss_val)
        log_["Validation Accuracy"].append(acc_val)
        print("Validation Loss:" + str(loss_val))
        print("Validation Accuracy:" + str(acc_val))
    return log_


def accuracy_score(_preds, _actuals):
    if (_actuals.shape[0] != _preds.shape[0]
            or _actuals.shape[1] != _preds.shape[1]
            or _actuals.shape[2] != _preds.shape[2]):
        print("Unmatched for accuracy_score")
        return 0
    abs = np.absolute(np.subtract(_preds, _actuals))
    err = np.sum(abs)
    acc = 1 - (err / (255 * len(_actuals) * len(_actuals[0]) * len(_actuals[0][0])))
    return acc


def evaluate(_dl, _model):
    accuracies = []
    losses = []
    for (i, (input, target)) in enumerate(_dl):
        output = _model(input)
        pred = output.squeeze().detach().numpy()
        actual = target.squeeze().detach().numpy()
        acc = accuracy_score(pred, actual)
        accuracies.append(acc)
        loss = loss_func(output, target).item()
        losses.append(loss)
    return sum(accuracies) / len(accuracies), sum(losses) / len(losses)


def visualize_log(_log):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(log["Epoch"], log["Training Loss"], '-r', label="Training")
    ax[0].plot(log["Epoch"], log["Validation Loss"], '-b', label="Validation")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("MSE Loss")
    ax[0].legend()
    ax[1].plot(log["Epoch"], log["Training Accuracy"], '-r', label="Training")
    ax[1].plot(log["Epoch"], log["Validation Accuracy"], '-b', label="Validation")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim([0.6, 1])
    ax[1].legend()
    diagram_func("result.png")


model = Model()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
log = train(trn_dl, val_dl, model, epochs)
visualize_log(log)
acc_tst, loss_tst = evaluate(tst_dl, model)
print("\nTesting:")
print("Testing Loss:" + str(loss_tst))
print("Testing Accuracy:" + str(acc_tst))
