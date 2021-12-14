# -*- coding: utf-8 -*-
import dlib
from faces_model import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


diagrams_dir = "./diagrams/" + os.path.basename(__file__).replace(".py", "")


print("Is this my face?")
print("Load pre-trained model...")
current_timestamp()
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, size, size, 3])
output = cnn_model(x, keep_prob_1, keep_prob_2)
predict = tf.argmax(output, 1)
saver = tf.train.Saver()  
session = tf.Session()
saver.restore(session, tf.train.latest_checkpoint('./model'))


def is_my_face(_img):
    res = session.run(predict, feed_dict={x: [_img / 255.0], keep_prob_1: 1.0, keep_prob_2: 1.0})
    return res[0] == 1


detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0)
print("Look at the camera.")
count = 0
correct = 0
limit = 40
flag = True
while flag:
    _, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = detector(img_gray, 1)
    if len(results) > 0:
        for (i, d) in enumerate(results):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = img[x1: y1, x2: y2]
            face = cv2.resize(face, (size, size))
            cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
            result = is_my_face(face)
            if result:
                correct += 1
            count += 1
            print("\r{0} {1}%".format(result, int(count / limit * 1000) / 10), end="")
            if count >= limit:
                flag = False
                break
    cv2.imshow("Look at the camera", img)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break
  
session.close()
if count >= limit:
    print("\rConfidence that this is my face: {0}%".format(int(correct / limit * 1000) / 10))
