# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

DATA_DIR = "data/CTImagesInCOVID19/"
DIAGRAM_DIR = "diagrams/" + os.path.basename(__file__).replace(".py", "")
SHOW_DIAGRAM = True
SAVE_DIAGRAM = True


def diagram_func(_filename):
    if not os.path.exists(DIAGRAM_DIR):
        os.makedirs(DIAGRAM_DIR)
    if SHOW_DIAGRAM:
        plt.show()
    if SAVE_DIAGRAM:
        plt.savefig(DIAGRAM_DIR + "/" + _filename, bbox_inches='tight')
        plt.close()


def load_image(_file):
    ct_img = nib.load(_file)
    return np.array(ct_img.dataobj)


def scan(_nib0bj):
    print(np.sum(_nib0bj < -2048))


# Loading
def load():
    print("\nLoading all computerized tomography images")
    files = os.listdir(DATA_DIR)
    for (i, file) in enumerate(files):
        if ".nii.gz" not in file:
            continue
        print("\rLoading process " + str(int(i / len(files) * 10000) / 100) + "% ...", end='')  # progress bar
        nibobjs = load_image(DATA_DIR + "/" + file)
        scan(nibobjs)
    print("\rLoading process 100.00% ...",)
    print("Finish loading")


# main
res = load()
