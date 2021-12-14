# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

DATA_DIR = "data/CT-Covid-19-August2020/"
FILE = "volume-covid19-A-0003.nii.gz"
DIAGRAM_DIR = "diagrams/analyze_data"
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

img = nib.load(DATA_DIR + FILE)
print(img)
print(img.header['db_name'])  # 输出头信息

# shape有四个参数 patient001_4d.nii.gz
# shape有三个参数 patient001_frame01.nii.gz   patient001_frame12.nii.gz
# shape有三个参数  patient001_frame01_gt.nii.gz   patient001_frame12_gt.nii.gz
width, height, queue = img.dataobj.shape
print(img.dataobj.shape)
print(type(img.dataobj))
print(img.dataobj[:, :, 0])
print(img.dataobj[:, :, 0].shape)
print(type(img.dataobj[0]))

num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
    if num > 20:
        break

diagram_func(FILE.replace(".nii.gz", ".png"))
