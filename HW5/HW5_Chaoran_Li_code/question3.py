# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
import cv2

SEED = 1
DIAGRAM_DIR = "diagrams/q3"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True

np.random.seed(SEED)
torch.manual_seed(SEED)

img_dir = "Q1.jpeg"
img_o = cv2.imread(img_dir, 0)

kernels = np.asarray([
    [[1, 2, 1],
     [0, 0, 0],
     [-1, -2, -1]],
    [[1, 0, -1],
     [2, 0, -2],
     [1, 0, -1]]
])  # horizontal, vertical
print("Kernels:")
for k in kernels:
    print(k)


def manual_edge_detect(_img, _kernel):
    r_offset = _kernel.shape[0] // 2
    c_offset = _kernel.shape[1] // 2
    outp = np.zeros(_img.shape, np.uint8)
    for r in range(_img.shape[0]):
        for c in range(_img.shape[1]):
            sum = 0
            for i in range(_kernel.shape[0]):
                for j in range(_kernel.shape[1]):
                    r_index = r - r_offset + i
                    c_index = c - c_offset + j
                    if 0 <= r_index < _img.shape[0] and 0 <= c_index < _img.shape[1]:
                        sum += _kernel[i][j] * _img[r - r_offset + i][c - c_offset + j]
            if sum < 0:
                sum = 0
            if sum > 255:
                sum = 255
            outp[r][c] = sum
    return outp


img_h = manual_edge_detect(img_o, kernels[0])
img_v = manual_edge_detect(img_o, kernels[1])
# combine to result
img_c = np.zeros(img_o.shape, np.uint8)
for r in range(img_o.shape[0]):
    for c in range(img_o.shape[1]):
        img_c[r][c] = max(img_h[r][c], img_v[r][c])

if SAVE_DIAGRAM:
    if not os.path.exists(DIAGRAM_DIR):
        os.makedirs(DIAGRAM_DIR)
    cv2.imwrite(DIAGRAM_DIR + "/original.jpeg", img_o)
    cv2.imwrite(DIAGRAM_DIR + "/vertical.jpeg", img_v)
    cv2.imwrite(DIAGRAM_DIR + "/horizontal.jpeg", img_h)
    cv2.imwrite(DIAGRAM_DIR + "/combined.jpeg", img_c)
if SHOW_DIAGRAM:
    cv2.imshow("original", img_o)
    cv2.imshow("vertical", img_v)
    cv2.imshow("horizontal", img_h)
    cv2.imshow("combined", img_c)
    cv2.waitKey(0)  # press enter to close the window
cv2.destroyAllWindows()

X = img_o / 255.0
y = img_c / 255.0


class ReshapedDataset(tud.Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32).reshape((1, X.shape[0], X.shape[1]))
        self.y = y.astype(np.float32).reshape((-1))
        print("Reshaped data:")
        print(self.X.shape, self.y.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X, self.y]


ds = ReshapedDataset(X, y)
dl = tud.DataLoader(ds)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv1.weight.data = torch.Tensor([[kernels[0].tolist()], [kernels[1].tolist()]])
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, 1, padding=1)

    def image_func(self, _X, _name):
        img = (_X * 255).detach().numpy().squeeze().reshape(2, img_o.shape[0], img_o.shape[1]).astype(np.uint8)
        if SAVE_DIAGRAM:
            if not os.path.exists(DIAGRAM_DIR):
                os.makedirs(DIAGRAM_DIR)
            cv2.imwrite(DIAGRAM_DIR + "/" + str(_name) + "_vertical.jpeg", img[0])
            cv2.imwrite(DIAGRAM_DIR + "/" + str(_name) + "_horizontal.jpeg", img[1])
        if SHOW_DIAGRAM:
            cv2.imshow(str(_name) + "-vertical", img[0])
            cv2.imshow(str(_name) + "-horizontal", img[1])
            cv2.waitKey(0)  # press enter to close the window
        cv2.destroyAllWindows()

    def forward(self, X):
        X = self.conv1(X)
        self.image_func(X, "conv1")
        X = self.act1(X)
        self.image_func(X, "act1")
        X = self.pool1(X)
        self.image_func(X, "pool1")
        return X


model = Model()
for i, (inputs, targets) in enumerate(dl):
    pred = model(inputs)
