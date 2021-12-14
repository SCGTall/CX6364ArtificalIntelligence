# -*- coding: utf-8 -*-
import numpy as np
import json
from faces_model import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


diagrams_dir = "./diagrams/" + os.path.basename(__file__).replace(".py", "")
learning_rate = 0.01  # [0.1, 0.01, 0.001, 0.0001, 0.00001]  learning rates
batch_size = 128  # [64, 96, 128, 192, 256]  batch sizes
conv_layer_num = 1  # [3, 2, 1]  3 -> cnn_model, 2 -> cnn_model2, 1 -> cnn_model3
epoches = 5


def train(_x_trn, _x_val, _y_trn, _y_val, _lr, _bs, _ln):
    print("Learning rate: {0}".format(_lr))
    print("Batch size: {0}".format(_bs))
    print("Convolution layer number: {0}".format(_ln))
    x = tf.placeholder(tf.float32, [None, size, size, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    keep_prob_1 = tf.placeholder(tf.float32)
    keep_prob_2 = tf.placeholder(tf.float32)
    batch_num = len(x_trn) // _bs
    if _ln == 2:
        out = cnn_model2(x, keep_prob_1, keep_prob_2)
    elif _ln == 1:
        out = cnn_model3(x, keep_prob_1, keep_prob_2)
    else:
        out = cnn_model(x, keep_prob_1, keep_prob_2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    train_step = tf.train.AdamOptimizer(_lr).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), tf.float32))
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    log_dic = {"Epoch": [],
               "Loss": [],
               "Accuracy": []}
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for ep in range(epoches):
            print("Epoch: {0}".format(ep), end="\t")
            log_dic["Epoch"].append(ep)
            current_timestamp()
            loss_mem = 0
            for b in range(batch_num):
                x_batched = _x_trn[b * _bs: (b + 1) * _bs]
                y_batched = _y_trn[b * _bs: (b + 1) * _bs]
                _, loss, summary = session.run([train_step, cross_entropy, merged_summary_op],
                                               feed_dict={x: x_batched,
                                                          y: y_batched,
                                                          keep_prob_1: 0.5,
                                                          keep_prob_2: 0.75})
                print("\rBatch Loss: {0}, {1}%".format(loss, int(b / batch_num * 10000) / 100), end="")
                loss_mem += loss
            log_dic["Loss"].append(float(loss_mem / batch_num))
            print("\rAverage Loss: {0}, 100.00%".format(loss_mem / batch_num))
            acc = accuracy.eval({x: _x_val,
                                 y: _y_val,
                                 keep_prob_1: 1,
                                 keep_prob_2: 1})
            log_dic["Accuracy"].append(float(acc))
            print("Validation Accuracy: {0}%".format(int(acc * 1000000) / 10000))
        return log_dic


def visualize_log(_outp_name, _log, _x, _y1="Loss", _y2="Accuracy"):
    if not os.path.exists(diagrams_dir):
        os.makedirs(diagrams_dir)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(_log[_x], _log[_y1], '-r')
    ax[0].set_xlabel(_x)
    ax[0].set_ylabel(_y1)
    ax[1].plot(_log[_x], _log[_y2], '-b')
    ax[1].set_xlabel(_x)
    ax[1].set_ylabel(_y2)
    ax[1].set_ylim([0.6, 1])
    plt.savefig(diagrams_dir + _outp_name, bbox_inches='tight')
    plt.close()


print("Load all faces.")
current_timestamp()
imgs = []
labels = []
imgs, labels = load_faces(my_faces_dir, imgs, labels)
imgs, labels = load_faces(other_faces_dir, imgs, labels)
imgs, labels = load_faces(friend_faces_dir, imgs, labels)
imgs = np.array(imgs)
labels = np.array(labels)
print(imgs.shape)
print(labels.shape)
x_trn, x_val, y_trn, y_val = train_test_split(imgs, labels, test_size=0.05, random_state=seed)
x_trn = np.divide(x_trn.astype('float32'), 255.0)
x_val = np.divide(x_val.astype('float32'), 255.0)
print("Training size: {0}".format(x_trn.shape[0]))
print("Testing size: {0}".format(x_val.shape[0]))


print("Tuning...")
log = train(x_trn, x_val, y_trn, y_val, _lr=learning_rate, _bs=batch_size, _ln=conv_layer_num)
#sub_name = "/lr_" + str(learning_rate).replace('.', '_')
#sub_name = "/bs_" + str(batch_size)
sub_name = "/ln_" + str(conv_layer_num)
img_name = sub_name + ".png"
visualize_log(img_name, _log=log, _x="Epoch")
with open(json_dir + sub_name + ".json", "w") as json_file:
    json.dump(log, json_file, indent=4)
