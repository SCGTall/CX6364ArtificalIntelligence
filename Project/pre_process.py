# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt
import nibabel as nib

DATA_DIR = "data/CTImagesInCOVID19/"
DIAGRAM_DIR = "diagrams/" + os.path.basename(__file__).replace(".py", "")
OUTPUT_DIR = "data/Processed/"
IMAGES_DIR = OUTPUT_DIR + "images/"
SINOGRAMS_DIR = OUTPUT_DIR + "sinograms/"
RECONSTRUCTIONS_DIR = OUTPUT_DIR + "reconstructions/"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True
SAVE_SINOGRAM = False


def check_folder(_folder):
    if not os.path.exists(_folder):
        os.makedirs(_folder)


def init():
    print("Input: " + DATA_DIR)
    print("Output: " + OUTPUT_DIR)
    check_folder(OUTPUT_DIR)
    check_folder(IMAGES_DIR)
    if SAVE_SINOGRAM:
        check_folder(SINOGRAMS_DIR)
    check_folder(RECONSTRUCTIONS_DIR)


def brief_check_image(_img):
    print(_img[::64, ::64])


def diagram_func(_filename):
    check_folder(DIAGRAM_DIR)
    if SHOW_DIAGRAM:
        plt.show()
    if SAVE_DIAGRAM:
        plt.savefig(DIAGRAM_DIR + "/" + _filename, bbox_inches='tight')
        plt.close()


def load_image(_file, _size):
    ct_img = nib.load(_file)
    nibobj = np.add(np.array(ct_img.dataobj), 2048)  # make outside from -2048 to 0 to make outside 0
    state = 0
    if np.sum(nibobj < 0) > 1000:
        state = 1
    if nibobj.shape[0] != _size[0] or nibobj.shape[1] != _size[1]:
        state = 2
    return (state, nibobj.clip(0, 3072))


"""
Available class:
"i": image
"s": sinogram
"r": recontruction
"""
def save_image(_img, _name, _class):
    if _class == "i":
        dir = IMAGES_DIR
    elif _class == "s":
        dir = SINOGRAMS_DIR
    elif _class == "r":
        dir = RECONSTRUCTIONS_DIR
    else:
        print("Unvalid class for save_image")
        return
    dir += name + "_" + _class + ".npy"
    np.save(dir, _img)
    plt.imshow(_img, cmap="gray")
    diagram_func(_name[-1] + "_" + _class + ".png")  # monitor


# Radon transforming
def radon_trans(_img, theta=360):
    sino = []
    for th in range(theta):
        tempgram = rotate(_img, angle=th, resize=False)
        projection = np.sum(tempgram, axis=0)
        sino.append(projection)
    outp = np.array(sino[::-1])
    return outp


# Iradon reconstruction
def iradon_trans(_sino):
    c = len(_sino)  # number of channels in sinogram
    s = 1  # step length during pack projection
    l = len(_sino[0])  # side length of image
    reconstruction = np.zeros((l, l))  # init with zeros
    for th in range(0, c, s):
        projection = _sino[th, :]
        # use expand_dims and repeat to deal with each slice
        expand = np.expand_dims(projection, axis=0)
        refill = expand.repeat(l, axis=0)
        rotated = rotate(refill, angle=th, resize=False)
        reconstruction += rotated
    return reconstruction.astype(int)


# main

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("Pre-processing all computerized tomography images")
init()
files = os.listdir(DATA_DIR)
count = 0
size = (512, 512)
print("Processing..")
for (i, file) in enumerate(files):
    if ".nii.gz" not in file:
        continue
    # load img from file
    (state, nibobjs) = load_image(DATA_DIR + "/" + file, size)
    if state != 0:
        continue
    print("\r" + file)
    for j in range(nibobjs.shape[2]):
        img = nibobjs[:, :, j]
        name = str(count).zfill(4)
        save_image(img, _name=name, _class="i")
        sino = radon_trans(img)
        if SAVE_SINOGRAM:
            save_image(sino, _name=name, _class="s")
        recon = iradon_trans(sino)
        save_image(recon, _name=name, _class="r")
        print("\rid: {0}, scaned {1}% files".format(name, int(i / len(files) * 10000) / 100), end='')  # progress bar
        count += 1
print("\rin total 100% finished.",)
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
