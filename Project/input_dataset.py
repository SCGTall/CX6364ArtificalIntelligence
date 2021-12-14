# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

DIAGRAM_DIR = "diagrams/" + os.path.basename(__file__).replace(".py", "")
OUTPUT_DIR = "data/short/"
IMAGES_DIR = OUTPUT_DIR + "images/"
SINOGRAMS_DIR = OUTPUT_DIR + "sinograms/"
RECONSTRUCTIONS_DIR = OUTPUT_DIR + "reconstructions/"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True


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
        print(_folder + file)
        img = np.load(_folder + file)
        name = int(file[:-6])
        memory[name] = img
    sorted_mem = sorted(memory.items(), key=lambda item:item[0])
    set = []
    for item in sorted_mem:
        set.append(item[1])
    return set


imgs = load_dataset(RECONSTRUCTIONS_DIR)
print(len(imgs))
for i in range(10):
    img = imgs[i]
    brief_check_image(img)
    plt.imshow(img, cmap="gray")
    diagram_func(str(i) + ".png")
