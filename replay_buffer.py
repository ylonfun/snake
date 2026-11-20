"""经验回放缓冲：预分配定长 NumPy 数组，循环覆盖。

随机采样 mini-batch，打破相邻 transition 的时序相关性。预分配比用 deque
存 Python 对象更省内存、采样更快。
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), np.float32)
        self.next_states = np.zeros((capacity, state_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.idx = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ids = np.random.randint(0, self.size, batch_size)
        return (self.states[ids], self.actions[ids], self.rewards[ids],
                self.next_states[ids], self.dones[ids])

    def __len__(self):
        return self.size
