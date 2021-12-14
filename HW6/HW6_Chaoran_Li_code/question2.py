# -*- coding: utf-8 -*-
# Use RMSProp as optimizer and MSELoss as loss function
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

DIAGRAM_DIR = "diagrams/q2"
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

terminals = [0, map_size[0] * map_size[1] - 1]

# features for neural network
features = [fin, f1, f2, f3, fout] = [q_size[0], 32, 8, 16, len(actions)]
hot = torch.eye(q_size[0])
alpha = 1e-3
iterations = 100
episodes = 15000
prob_exploration = 0.1


class Model(nn.Module):
    def __init__(self, _gamma=0.7, _iterations=100, _episodes=15000, _prob_exploration=0.1):
        super(Model, self).__init__()
        self.gamma = _gamma
        self.iterations = _iterations
        self.episodes = _episodes
        self.prob_exploration = _prob_exploration
        self.fc1 = nn.Linear(fin, f1)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(f1, f2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(f2, f3)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(f3, fout)

    def forward(self, X):
        X = self.fc1(X)
        X = self.act1(X)
        X = self.fc2(X)
        X = self.act2(X)
        X = self.fc3(X)
        X = self.act3(X)
        X = self.fc4(X)
        return X


def train(model):
    print("\nTraining...")
    paths = []
    cnt_init_s = np.zeros(q_size[0])
    for ep in range(model.episodes):
        s = np.random.randint(0, q_size[0])
        while s in terminals:
            s = np.random.randint(0, q_size[0])
        cnt_init_s[s] += 1
        path = [s]
        for i in range(model.iterations):
            q_values = model(hot[s])
            if np.random.rand() < model.prob_exploration:
                a = np.random.randint(0, len(actions))
                q = q_values[a]
            else:
                q, a_t = torch.max(q_values, 0)
                a = a_t.item()  # tensor to int

            # configure next state
            shift = actions[a]
            if ((s % map_size[1] == 0 and a == 1)
                    or ((s + 1) % map_size[1] == 0 and a == 0)
                    or (np.floor(s / map_size[1]) == 0 and a == 2)
                    or (np.floor(s / map_size[1]) == map_size[0] - 1 and a == 3)):
                shift = 0
            s_next = s + shift
            s_next_in_terminals = (s_next in terminals)

            # get target Q value
            with torch.no_grad():
                q_values_next = model(hot[s_next])
                q_next = torch.max(q_values_next)
            q_target = torch.tensor(reward, dtype=torch.float32) + (
                model.gamma * q_next if not s_next_in_terminals else 0)

            loss = loss_func(q, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            path.append(s_next)

            if s_next in terminals:
                break

            s = s_next

        paths.append(path)


def evaluate(model):
    print("\nEvaluating...")
    print("Paths:")
    for s in range(map_size[0] * map_size[1]):
        path = [s]
        fg_loop = False
        while s not in terminals:
            with torch.no_grad():
                q, a_t = torch.max(model(hot[s]), 0)
                a = a_t.item()

            shift = actions[a]
            if (s % map_size[1] == 0 and a == 1) or \
                    ((s + 1) % map_size[1] == 0 and a == 0) or \
                    (np.floor(s / map_size[1]) == 0 and a == 2) or \
                    (np.floor(s / map_size[1]) == map_size[0] - 1 and a == 3):
                break
            s += shift

            if s in path:
                fg_loop = True
                break
            path.append(s)

        print(path, "Loop detected" if fg_loop else "")


def visualize(model, outp):
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

    for i in range(map_size[0] * map_size[1]):
        r = map_size[0] - np.floor(i / map_size[1]) - 0.5
        c = i % map_size[1] + 0.5
        if i in terminals:
            plt.arrow(c - 0.25, r - 0.25, 0.5, 0.5, head_width=0, color="r")
            plt.arrow(c + 0.25, r - 0.25, -0.5, 0.5, head_width=0, color="r")
            continue

        with torch.no_grad():
            q_values = model(hot[i])
            q, j = torch.max(q_values, 0)
            dx = len_arrow_tail if j == 0 else -len_arrow_tail if j == 1 else 0
            dy = len_arrow_tail if j == 2 else -len_arrow_tail if j == 3 else 0
            plt.arrow(c, r, dx, dy, head_width=len_arrow_head, color=color_arrow)

    diagram_func(outp)


model = Model(_gamma=gamma, _iterations=iterations, _episodes=episodes, _prob_exploration=prob_exploration)
loss_func = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha)

print("\nBefore training:")
with torch.no_grad():
    for i in range(map_size[0] * map_size[1]):
        print(model(hot[i]))

train(model)
evaluate(model)
visualize(model, "result.png")

print("\nAfter training:")
with torch.no_grad():
    for i in range(map_size[0] * map_size[1]):
        print(model(hot[i]))
