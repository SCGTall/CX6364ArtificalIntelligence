# -*- coding: utf-8 -*-
import os
import cv2
import dlib

input_dir = "./data/lfw"
output_dir = "./data/other_faces"
size = 64
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
detector = dlib.get_frontal_face_detector()
index = 1
for (root, dirs, files) in os.walk(input_dir):
    for file in files:
        if file.endswith('.jpg'):
            img_dir = root + "/" + file
            img = cv2.imread(img_dir)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = detector(img_gray, 1)
            for i, d in enumerate(results):
                print("Face No. {0}".format(index))
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1: y1, x2: y2]
                face = cv2.resize(face, (size, size))
                if index % 100 == 1:
                    cv2.imshow("Face No. {0}".format(index), face)
                cv2.imwrite(output_dir + "/" + str(index).zfill(5) + '.png', face)
                index += 1
            key = cv2.waitKey(30) & 0xff  # only consider last 8 digit in ASCII
            if key == 27:
                break
print('Finished!')
