# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib

DATA_DIR = "data/short/"
DIAGRAM_DIR = "diagrams/" + os.path.basename(__file__).replace(".py", "")

def load_image(_file):
    ct_img = nib.load(_file)
    nibobj = np.add(np.array(ct_img.dataobj), 2048)  # make outside from -2048 to 0 to make outside 0
    if np.sum(nibobj < 0) > 1000:
        return None
    else:
        return nibobj.clip(0, 3060)


def brief_check_image(_img):
    print(_img[::64, ::64])


files = os.listdir(DATA_DIR)
shape = [0, 512, 512]
id = 0
if not os.path.exists(DIAGRAM_DIR):
    os.makedirs(DIAGRAM_DIR)
targets = []
imgs1 = []
imgs2 = []
for (i, file) in enumerate(files):
    if ".nii.gz" not in file:
        continue
    # load img from file
    nibobjs = load_image(DATA_DIR + "/" + file)
    for i in range(nibobjs.shape[2]):
        img = nibobjs[:, :, i]
        name = str(id).zfill(4)
        np.save(DIAGRAM_DIR + "/" + name + ".npy", img)
        if i <= 4:
            brief_check_image(img)
            targets.append(DIAGRAM_DIR + "/" + name + ".npy")
            imgs1.append(img)
        id += 1

print("####")
for dir in targets:
    img = np.load(dir)
    brief_check_image(img)
    imgs2.append(img)

for i in range(4):
    print(np.sum(np.subtract(imgs1[i], imgs2[i])))

print(type(imgs1[0]))
