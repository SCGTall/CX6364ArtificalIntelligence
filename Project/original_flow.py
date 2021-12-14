# -*- coding: utf-8 -*-
import os
import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt
import nibabel as nib

DATA_DIR = "data/CTImagesInCOVID19short/"
DIAGRAM_DIR = "diagrams/original_flow"
SHOW_DIAGRAM = False
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
    #print(_file)
    #print(img.dataobj.shape)
    nibobj = np.add(np.array(ct_img.dataobj), 2048)  # make outside from -2048 to 0 to make outside 0
    if np.sum(nibobj < 0) > 1000:
        return None
    else:
        return nibobj.clip(0, 3072)


def append_list(_list, _nib0bj):
    if len(_nib0bj) == None:
        return _list
    if len(_list) > 0 and _list[0].shape != _nib0bj[:, :, 0].shape:
        print("Different input in append_list")
        return _list
    for i in range(_nib0bj.shape[2]):
        _list.append(_nib0bj[:, :, i])
    return _list


# Loading
def load():
    print("\nLoading all computerized tomography images")
    files = os.listdir(DATA_DIR)
    imgs = []
    for (i, file) in enumerate(files):
        if ".nii.gz" not in file:
            continue
        print("\rLoading process " + str(int(i / len(files) * 10000) / 100) + "% ...", end='')  # progress bar
        nibobjs = load_image(DATA_DIR + "/" + file)
        imgs = append_list(imgs, nibobjs)
    print("\rLoading process 100.00% ...",)
    print("Shape: ({0}, {1}, {2})".format(len(imgs), imgs[0].shape[0], imgs[0].shape[1]))
    print("Finish loading")
    return imgs


def radon_trans(_img, theta=360):
    sino = []
    for th in range(theta):
        tempgram = rotate(_img, angle=th, resize=False)
        projection = np.sum(tempgram, axis=0)
        sino.append(projection)
    outp = np.array(sino[::-1])
    return outp


def show_sino(_sino, _name):
    #print("Show sinograms: " + _name)
    dx = 1. / _sino.shape[1]
    dy = 1.
    plt.figure(figsize=(4, 6))
    plt.title("Rotate and sum on columns\n(Sinogram)")
    plt.xlabel("Projection position")
    plt.ylabel("Projection angle")
    plt.imshow(_sino,
               cmap="gray",
               extent=(-dx, _sino.shape[1] + dx, -dy, 360. + dy),
               aspect='auto')
    plt.yticks(range(0, 361, 60))  # show 360 degree on top
    diagram_func(_name)


# Radon transforming
def transform(_imgs):
    print("\nTransform images to sinograms")
    sinograms = []
    for (i, img) in enumerate(_imgs):
        print("\rTransforming " + str(int(i / len(_imgs) * 10000) / 100) + "% ...", end='')  # progress bar
        sino = radon_trans(img)
        sinograms.append(sino)
    print("\rTransforming 100.00% ...",)
    print("Shape: ({0}, {1}, {2})".format(len(sinograms), sinograms[0].shape[0], sinograms[0].shape[1]))
    print("Finish transforming")
    return sinograms


# Iradon reconstruction
def iradon_trans(_sino):
    c = len(_sino)  # number of channels in sinogram
    s = 1  # step length during pack projection
    w = 1.0 / np.floor(c / s)  # weigth for each slice
    l = len(_sino[0])  # side length of image
    reconstruction = np.zeros((l, l))  # init with zeros
    for th in range(0, c, s):
        projection = _sino[th, :]
        # use expand_dims and repeat to deal with each slice
        expand = np.expand_dims(projection, axis=0)
        refill = expand.repeat(l, axis=0)
        rotated = rotate(refill, angle=th, resize=False)
        reconstruction += w * rotated
    return reconstruction


def reconstruct(_sinos):
    print("\nReconstructing images from sinograms")
    reconstructions = []
    for (i, sino) in enumerate(_sinos):
        print("\rReconstructing " + str(int(i / len(_sinos) * 10000) / 100) + "% ...", end='')  # progress bar
        recon = iradon_trans(sino)
        reconstructions.append(recon)
    print("\rReconstructing 100.00% ...", )
    print("Shape: ({0}, {1}, {2})".format(len(reconstructions),
                                          reconstructions[0].shape[0],
                                          reconstructions[0].shape[1]))
    print("Finish reconstructing")
    return reconstructions


# main
imgs = load()
sinograms = transform(imgs)
reconstructions = reconstruct(sinograms, None)
for i in range(1, 7):
    img_name = "img" + str(i) + ".png"
    plt.imshow(imgs[i * 5], cmap="gray")
    diagram_func(img_name)
    sino_name = "sinogram" + str(i) + ".png"
    show_sino(sinograms[i * 5], sino_name)
    recon_name = "recontruction" + str(i) + ".png"
    plt.imshow(reconstructions[i * 5], cmap="gray")
    diagram_func(recon_name)
    diff_name = "difference" + str(i) + ".png"
    diff = imgs[i * 5] - reconstructions[i * 5]
    plt.imshow(diff, cmap="gray")
    diagram_func(diff_name)