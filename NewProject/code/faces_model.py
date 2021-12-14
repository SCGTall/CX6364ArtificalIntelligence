# -*- coding: utf-8 -*-
import cv2
import os
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


my_faces_dir = "./data/my_faces"
other_faces_dir = "./data/other_faces"
friend_faces_dir = "./data/friends_faces"
json_dir = "./data/json"
model_dir = "./model"
seed = 43
size = 64
batch_size = 128
learning_rate = 0.01
epoches = 10


def current_timestamp():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def get_padding_size(_img):
    h, w, _ = _img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    bigger = max(h, w)
    if w < bigger:
        left = (bigger - w) // 2
        right = (bigger - w) - left
    elif h < bigger:
        top = (bigger - h) // 2
        bottom = (bigger - h) - top
    else:
        pass
    return top, bottom, left, right


def load_faces(_dir, _imgs, _labels, h=size, w=size):
    print("Load faces from " + _dir)
    for file in os.listdir(_dir):
        if file.endswith('.png'):
            img = cv2.imread(_dir + "/" + file)
            top, bottom, left, right = get_padding_size(img)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            _imgs.append(img)
            _labels.append([0, 1] if _dir == my_faces_dir else [1, 0])
    return _imgs, _labels


# models
def weight_variable(_shape):
    init = tf.random_normal(_shape, stddev=0.01)
    return tf.Variable(init)


def bias_variable(_shape):
    init = tf.random_normal(_shape)
    return tf.Variable(init)


def conv2d(_x, _w):
    return tf.nn.conv2d(_x, _w, strides=[1, 1, 1, 1], padding='SAME')  # (batch, width, height, channel)


def conv2d2(_x, _w):
    return tf.nn.conv2d(_x, _w, strides=[1, 2, 2, 1], padding='SAME')


def maxpool(_x):
    return tf.nn.max_pool(_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def maxpool2(_x):
    return tf.nn.max_pool(_x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


def dropout(_x, _keep):
    return tf.nn.dropout(_x, _keep)


def cnn_model(_inp, _kp1, _kp2):
    w1 = weight_variable([3, 3, 3, 32])
    b1 = bias_variable([32])
    conv1 = tf.nn.relu(conv2d(_inp, w1) + b1)
    pool1 = maxpool(conv1)
    drop1 = dropout(pool1, _kp1)

    w2 = weight_variable([3, 3, 32, 64])
    b2 = bias_variable([64])
    conv2 = tf.nn.relu(conv2d(drop1, w2) + b2)
    pool2 = maxpool(conv2)
    drop2 = dropout(pool2, _kp1)

    w3 = weight_variable([3, 3, 64, 64])
    b3 = bias_variable([64])
    conv3 = tf.nn.relu(conv2d(drop2, w3) + b3)
    pool3 = maxpool(conv3)
    drop3 = dropout(pool3, _kp1)

    w4 = weight_variable([8 * 8 * 64, 512])
    b4 = bias_variable([512])
    flat4 = tf.reshape(drop3, [-1, 8 * 8 * 64])
    dense = tf.nn.relu(tf.matmul(flat4, w4) + b4)
    drop4 = dropout(dense, _kp2)

    w5 = weight_variable([512, 2])
    b5 = bias_variable([2])
    outp = tf.add(tf.matmul(drop4, w5), b5)
    return outp


def cnn_model2(_inp, _kp1, _kp2):
    w1 = weight_variable([3, 3, 3, 32])
    b1 = bias_variable([32])
    conv1 = tf.nn.relu(conv2d(_inp, w1) + b1)
    pool1 = maxpool(conv1)
    drop1 = dropout(pool1, _kp1)

    w2 = weight_variable([3, 3, 32, 64])
    b2 = bias_variable([64])
    conv2 = tf.nn.relu(conv2d(drop1, w2) + b2)
    pool2 = maxpool2(conv2)
    drop2 = dropout(pool2, _kp1)

    w4 = weight_variable([8 * 8 * 64, 512])
    b4 = bias_variable([512])
    flat4 = tf.reshape(drop2, [-1, 8 * 8 * 64])
    dense = tf.nn.relu(tf.matmul(flat4, w4) + b4)
    drop4 = dropout(dense, _kp2)

    w5 = weight_variable([512, 2])
    b5 = bias_variable([2])
    outp = tf.add(tf.matmul(drop4, w5), b5)
    return outp

def cnn_model3(_inp, _kp1, _kp2):
    w1 = weight_variable([3, 3, 3, 64])
    b1 = bias_variable([64])
    conv1 = tf.nn.relu(conv2d2(_inp, w1) + b1)
    pool1 = maxpool2(conv1)
    drop1 = dropout(pool1, _kp1)

    w4 = weight_variable([8 * 8 * 64, 512])
    b4 = bias_variable([512])
    flat4 = tf.reshape(drop1, [-1, 8 * 8 * 64])
    dense = tf.nn.relu(tf.matmul(flat4, w4) + b4)
    drop4 = dropout(dense, _kp2)

    w5 = weight_variable([512, 2])
    b5 = bias_variable([2])
    outp = tf.add(tf.matmul(drop4, w5), b5)
    return outp
