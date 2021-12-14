# -*- coding: utf-8 -*-
import random
from faces_model import *
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

diagrams_dir = "./diagrams/" + os.path.basename(__file__).replace(".py", "")
demo_dir = "./data/demo"
me_dir = demo_dir + "/me"
not_me_dir = demo_dir + "/not_me"

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


def demo(_tag, _dir):
    count = 0
    correct = 0
    limit = 1000
    print("\nLoad faces from " + _tag)
    current_timestamp()
    faces = os.listdir(_dir)
    while count < limit:
        index = random.randint(0, len(faces) - 1)
        file = faces[index]
        if file.endswith('.png'):
            face = cv2.imread(_dir + "/" + file)
            top, bottom, left, right = get_padding_size(face)
            face = cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            face = cv2.resize(face, (size, size))
            result = is_my_face(face)
            if result:
                correct += 1
            count += 1
            print("\r{0} {1}%".format(result, int(count / limit * 1000) / 10), end="")
    if count >= limit:
        print("\rConfidence that this is my face: {0}%".format(int(correct / limit * 1000) / 10))


demo("me", me_dir)
demo("not_me", not_me_dir)
