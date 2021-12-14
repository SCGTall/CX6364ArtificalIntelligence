# -*- coding: utf-8 -*-
import cv2
import dlib
import os
import random
import numpy as np


def process_image(_img, _light=1., _bias=0.):
    processed = np.add(np.multiply(_img.astype(int), _light), _bias)  # val * _light + _bias
    return processed.clip(0, 255)


output_dir = "./data/my_faces"
size = 64
start_num = 1
end_num = 1000
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0)
print("Start collecting images from {0} to {1}.\nSmile :-)".format(start_num, end_num))
for index in range(start_num, end_num+1):
    success, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = detector(img_gray, 1)
    for (i, d) in enumerate(results):
        print("Face No. {0}".format(index))
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1: y1, x2: y2]
        # adjust image to make it random
        face = process_image(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
        face = cv2.resize(face, (size, size))
        if index % 100 == 1:
            cv2.imshow("Face No. {0}".format(index), face)
        cv2.imwrite(output_dir + "/" + str(index).zfill(5) + '.png', face)
    key = cv2.waitKey(30) & 0xff  # only consider last 8 digit in ASCII
    if key == 27:
        break
print('Finished!')
