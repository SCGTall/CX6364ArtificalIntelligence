# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

SEED = 1
np.random.seed(SEED)

DIAGRAM_DIR = "diagrams/q1"
SHOW_DIAGRAM = False
SAVE_DIAGRAM = True


def diagram_func(filename):
    if not os.path.exists(DIAGRAM_DIR):
        os.makedirs(DIAGRAM_DIR)
    if SHOW_DIAGRAM:
        plt.show()
    if SAVE_DIAGRAM:
        plt.savefig(DIAGRAM_DIR + "/" + filename, bbox_inches='tight')
        plt.close()


map_size = (5, 5)  # height, width
actions = {0: 1, 1: -1, 2: -map_size[1], 3: map_size[1]}  # right, left, up, down, k -> v: action -> shift
q_size = [map_size[0] * map_size[1], len(actions)]
gamma = 0.7
reward = -5
epsilon = 1e-7

cost_init_max = -10
terminals = [0, map_size[0] * map_size[1] - 1]
q = np.random.uniform(cost_init_max, 0, q_size)
q[0][:] = 0
q[-1][:] = 0

print(q)

next = q.copy()
flag = True

# get Q table
print("Get Q table")
round = 0
while flag:
    round += 1
    print("Round " + str(round) + ":")
    flag = False
    for i in range(1, q_size[0] - 1):  # traverse i out of terminals
        for j in range(q_size[1]):
            shift = actions[j]
            if ((i % map_size[1] == 0 and j == 1)
                    or ((i + 1) % map_size[1] == 0 and j == 0)
                    or (np.floor(i / map_size[1]) == 0 and j == 2)
                    or (np.floor(i / map_size[1]) == map_size[0] - 1 and j == 3)):
                shift = 0
            tmp = reward + gamma * np.max(q[i + shift])
            if abs(tmp - q[i][j]) > epsilon:
                next[i][j] = tmp
                flag = True

    q = next.copy()
    print(q)
print("End after " + str(round) + " rounds.")

# visualize

ax = plt.gca()
ax.grid(True)
plt.axis([0, map_size[1], 0, map_size[0]])
plt.xticks(np.arange(0, map_size[1], step=1))
plt.yticks(np.arange(0, map_size[0], step=1))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid(True, 'both', 'both', linewidth=2)

len_arrow_tail = 0.15
len_arrow_head = 0.15
color_arrow = "blue"

for i in range(q_size[0]):
    r = map_size[0] - np.floor(i / map_size[1]) - 0.5
    c = i % map_size[1] + 0.5
    if i in terminals:
        plt.arrow(c - 0.25, r - 0.25, 0.5, 0.5, head_width=0, color="r")
        plt.arrow(c + 0.25, r - 0.25, -0.5, 0.5, head_width=0, color="r")
        continue

    r_max = np.max(q[i])
    for j in range(len(q[i])):
        if abs(r_max - q[i][j]) < epsilon:
            dx = len_arrow_tail if j == 0 else -len_arrow_tail if j == 1 else 0
            dy = len_arrow_tail if j == 2 else -len_arrow_tail if j == 3 else 0
            plt.arrow(c, r, dx, dy, head_width=len_arrow_head, color=color_arrow)

diagram_func("result.png")