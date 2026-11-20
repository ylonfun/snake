"""DQN 智能体：Q 网络 + 目标网络、ε-greedy 探索、TD 学习、目标软更新。"""
import numpy as np

from base_agent import BaseAgent
from q_network import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    def __init__(self, state_mode="local", n_actions=None, net_sizes=None,
                 lr=0.001, gamma=0.95, buffer_capacity=100000, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=80000,
                 target_tau=0.005, learn_start=1000, train_freq=1,
                 huber_delta=1.0):
        if state_mode == "full":
            state_dim, default_sizes, default_actions = 300, (300, 64, 32, 4), 4
        elif state_mode == "energy":
            state_dim, default_sizes, default_actions = 43, (43, 64, 64, 5), 5
        elif state_mode == "energy_move":
            state_dim, default_sizes, default_actions = 43, (43, 64, 64, 4), 4
        elif state_mode == "survival":
            state_dim, default_sizes, default_actions = 43, (43, 64, 64, 5), 5
        else:
            state_dim, default_sizes, default_actions = 40, (40, 64, 64, 4), 4
        if net_sizes is None:
            net_sizes = default_sizes
        if n_actions is None:
            n_actions = default_actions
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.tau = target_tau
        self.learn_start = learn_start
        self.train_freq = train_freq
        self.q = QNetwork(list(net_sizes), lr, huber_delta)
        self.q_target = QNetwork(list(net_sizes), lr, huber_delta)
        self.sync_target(hard=True)
        self.buffer = ReplayBuffer(buffer_capacity, state_dim)
        self.step_count = 0

    def sync_target(self, hard=False):
        for i in range(len(self.q.W)):
            if hard:
                self.q_target.W[i] = self.q.W[i].copy()
                self.q_target.b[i] = self.q.b[i].copy()
            else:
                self.q_target.W[i] = ((1 - self.tau) * self.q_target.W[i]
                                      + self.tau * self.q.W[i])
                self.q_target.b[i] = ((1 - self.tau) * self.q_target.b[i]
                                      + self.tau * self.q.b[i])

    def update_eps(self):
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * frac

    def _fit(self, state):
        """把状态裁到本网络输入维度。

        local 环境现在产出 42 维（在旧 40 维后追加了 visit_count 与 scent），而
        现有 40 维 checkpoint 只吃前 40 维，且这 40 维布局与旧版逐位一致。其他
        模式（full 300 / energy 43 / survival 43）维度本就匹配，不做裁剪。
        """
        state = np.asarray(state, np.float32)
        return state[:self.state_dim] if state.shape[-1] > self.state_dim else state

    def act(self, state, training=True):
        if training:
            self.update_eps()
            if np.random.random() < self.eps:
                return np.random.randint(self.n_actions)
        qvals = self.q.forward(self._fit(state)[None])
        return int(qvals.argmax(1)[0])

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(self._fit(state), action, reward,
                         self._fit(next_state), done)

    def learn(self):
        self.step_count += 1
        if (self.buffer.size < self.learn_start
                or self.step_count % self.train_freq != 0):
            return None
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        # TD 目标：r + γ(1-done) max_a' Q_target(s')
        next_max = self.q_target.forward(s2).max(1)
        td = r + self.gamma * (1 - done) * next_max
        target = self.q.forward(s).copy()   # 非选中动作保持原值，梯度为 0
        target[np.arange(self.batch_size), a] = td
        self.q.update(s, target)
        self.sync_target()
        return float(td.mean())
