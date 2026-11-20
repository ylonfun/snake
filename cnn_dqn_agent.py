"""CNN-DQN 智能体：完整 grid 表征 + 卷积 Q 网络，目标网络 + ε-greedy + TD。

grid (6,10,10) 压成 600 维存入 ReplayBuffer，采样后还原空间形状喂 CNN。
目标网络参数按相同数组顺序做硬/软同步。
"""
import numpy as np

from base_agent import BaseAgent
from replay_buffer import ReplayBuffer
from cnn_qnet import CNNQNet
from conv import Conv2D

GRID_FLAT = 600
GRID_SHAPE = (-1, 6, 10, 10)


def _all_params(net):
    out = []
    for layer in net._learnable():
        if isinstance(layer, Conv2D):
            out += [layer.W, layer.b]
        else:  # MLP
            for i in range(len(layer.W)):
                out += [layer.W[i], layer.b[i]]
    return out


class CNNDQNAgent(BaseAgent):
    def __init__(self, lr=1e-3, gamma=0.95, buffer_capacity=100000, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=80000,
                 target_tau=0.005, learn_start=1000, huber_delta=1.0):
        self.n_actions = 4
        self.gamma, self.batch_size = gamma, batch_size
        self.eps = eps_start
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.tau, self.learn_start = target_tau, learn_start
        self.q = CNNQNet(lr=lr, huber_delta=huber_delta)
        self.q_target = CNNQNet(lr=lr, huber_delta=huber_delta)
        self.sync_target(hard=True)
        self.buffer = ReplayBuffer(buffer_capacity, GRID_FLAT)
        self.step_count = 0

    def sync_target(self, hard=False):
        src, dst = _all_params(self.q), _all_params(self.q_target)
        for s, d in zip(src, dst):
            if hard:
                d[:] = s
            else:
                d *= (1 - self.tau)
                d += self.tau * s

    def update_eps(self):
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, state, training=True):
        if training:
            self.update_eps()
            if np.random.random() < self.eps:
                return np.random.randint(self.n_actions)
        qv = self.q.forward(np.asarray([state], np.float32))
        return int(qv.argmax(1)[0])

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state.reshape(-1), action, reward,
                         next_state.reshape(-1), done)

    def learn(self):
        self.step_count += 1
        if self.buffer.size < self.learn_start:
            return None
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        s, s2 = s.reshape(GRID_SHAPE), s2.reshape(GRID_SHAPE)
        next_max = self.q_target.forward(s2).max(1)
        td = r + self.gamma * (1 - done) * next_max
        target = self.q.forward(s).copy()
        target[np.arange(self.batch_size), a] = td
        self.q.update(s, target)
        self.sync_target()
        return float(td.mean())

    def save(self, path):
        params = _all_params(self.q)
        np.savez(path, step=self.step_count,
                 **{f"p{i}": p for i, p in enumerate(params)})

    def load(self, path):
        data = np.load(path)
        for i, p in enumerate(_all_params(self.q)):
            p[:] = data[f"p{i}"]
        if "step" in data:
            self.step_count = int(data["step"])
        self.sync_target(hard=True)
