# -*- coding: utf-8 -*-
import numpy as np
from faces_model import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


diagrams_dir = "./diagrams/" + os.path.basename(__file__).replace(".py", "")


def train(_x_trn, _x_val, _y_trn, _y_val):
    x = tf.placeholder(tf.float32, [None, size, size, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    keep_prob_1 = tf.placeholder(tf.float32)
    keep_prob_2 = tf.placeholder(tf.float32)
    out = cnn_model(x, keep_prob_1, keep_prob_2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), tf.float32))
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    log_dic = {"Epoch": [],
               "Loss": [],
               "Accuracy": []}
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        round_num = 0
        summary_writer = tf.summary.FileWriter(model_dir + "/tmp", graph=tf.get_default_graph())
        for ep in range(epoches):
            print("Epoch: {0}".format(ep), end="\t")
            log_dic["Epoch"].append(ep)
            current_timestamp()
            loss_mem = 0
            for b in range(batch_num):
                x_batched = _x_trn[b * batch_size: (b + 1) * batch_size]
                y_batched = _y_trn[b * batch_size: (b + 1) * batch_size]
                _, loss, summary = session.run([train_step, cross_entropy, merged_summary_op],
                                               feed_dict={x: x_batched,
                                                          y: y_batched,
                                                          keep_prob_1: 0.5,
                                                          keep_prob_2: 0.75})
                round_num = ep * batch_num + b
                summary_writer.add_summary(summary, round_num)
                print("\rBatch Loss: {0}, {1}%".format(loss, int(b / batch_num * 10000) / 100), end="")
                loss_mem += loss
            log_dic["Loss"].append(loss_mem / batch_num)
            print("\rAverage Loss: {0}, 100.00%".format(loss_mem / batch_num))
            acc = accuracy.eval({x: _x_val,
                                 y: _y_val,
                                 keep_prob_1: 1,
                                 keep_prob_2: 1})
            log_dic["Accuracy"].append(acc)
            print("Validation Accuracy: {0}%".format(int(acc * 1000000) / 10000))
        saver.save(session, model_dir + "/train_faces.model", global_step=round_num)
        return log_dic


def visualize_log(_log):
    if not os.path.exists(diagrams_dir):
        os.makedirs(diagrams_dir)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(_log["Epoch"], _log["Loss"], '-r')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].plot(_log["Epoch"], _log["Accuracy"], '-b')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim([0.6, 1])
    plt.savefig(diagrams_dir + "/result.png", bbox_inches='tight')
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
batch_num = len(x_trn) // batch_size

print("Training...")
log = train(x_trn, x_val, y_trn, y_val)
visualize_log(log)
