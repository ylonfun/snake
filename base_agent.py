"""智能体抽象接口：训练器只依赖这个接口，不绑定具体算法。

DQNAgent 现在实现它；未来 WorldModelAgent 实现同一接口即可接入同一训练器。
"""


class BaseAgent:
    def act(self, state, training=True):
        raise NotImplementedError

    def remember(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
