# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import matplotlib.pyplot as plt

DIAGRAM_DIR = "diagrams/" + os.path.basename(__file__).replace(".py", "")
OUTPUT_DIR = "data/Processed/"
IMAGES_DIR = OUTPUT_DIR + "images/"
RECONSTRUCTIONS_DIR = OUTPUT_DIR + "reconstructions/"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True


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


def load_dataset(_folder):
    files = os.listdir(_folder)
    memory = {}
    for file in files:
        if ".npy" not in file:
            continue
        img = np.load(_folder + file)
        name = int(file[:-6])
        memory[name] = img
    sorted_mem = sorted(memory.items(), key=(lambda item: item[0]))
    set = []
    for item in sorted_mem:
        set.append(item[1])
    return set


def normalize_dataset(_ds, _class, _size=-1):
    size = _size
    if _size <= 0:
        size = len(_ds)
    maximum = 0
    minimum = 10000
    for arr in _ds:
        maximum = max(maximum, np.max(arr))
        minimum = min(minimum, np.min(arr))
    outp = []
    for (i, arr) in enumerate(_ds):
        if i >= size:
            break
        normed = np.divide(np.subtract(arr, minimum), maximum / 255).astype(np.int16)
        outp.append(normed)
        if i % 1000 == 0:
            print(i)
            brief_check_image(normed)
            plt.imshow(normed, cmap="gray")
            diagram_func(str(i // 1000) + "_" + _class + ".png")
    return np.asarray(outp)


current_timestamp()
dataset_size = -1
print("Prepare data:")
if dataset_size > 0:
    print("Dataset size: " + str(dataset_size))
imgs = load_dataset(IMAGES_DIR)
imgs_normed = normalize_dataset(imgs, "i", dataset_size)
file_name = "images.npy"
print("Saving " + file_name)
np.save(OUTPUT_DIR + "/" + file_name, imgs_normed)
recons = load_dataset(RECONSTRUCTIONS_DIR)
recons_normed = normalize_dataset(recons, "r", dataset_size)
file_name = "reconstructions.npy"
print("Saving " + file_name)
np.save(OUTPUT_DIR + "/" + file_name, recons_normed)

