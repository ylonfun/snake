"""保分辨率的 grid 前向世界模型（W5 重设计）。

不做潜在压缩、不池化：动作 one-hot 广播成 4 通道与 6 通道 grid 拼接，全卷积
保持 10×10 分辨率，直接预测 next grid；另一个卷积头经全局平均预测终止概率。
奖励不在此学习——由与环境一致的确定性规则从 (grid, next_grid, done) 计算，
避免"只从单状态预测 transition 奖励"的结构缺陷与回归欠拟合。
"""
import numpy as np

from conv import Conv2D, ReLU

DTYPE = np.float32


def _action_channels(action):
    n = len(action)
    out = np.zeros((n, 4, 10, 10), DTYPE)
    a = np.asarray(action).reshape(-1)
    out[np.arange(n), a] = 1.0
    return out


def _sigmoid(x):
    xc = np.clip(x, -60, 60)
    return np.where(xc >= 0, 1.0 / (1.0 + np.exp(-xc)),
                    np.exp(xc) / (1.0 + np.exp(xc)))


class GridForwardModel:
    def __init__(self, lr=3e-4):
        self.c1 = Conv2D(10, 32, 3, pad=1, lr=lr)
        self.r1 = ReLU()
        self.c2 = Conv2D(32, 32, 3, pad=1, lr=lr)
        self.r2 = ReLU()
        self.grid_head = Conv2D(32, 6, 3, pad=1, lr=lr)
        self.done_head = Conv2D(32, 1, 3, pad=1, lr=lr)

    def forward(self, grid, action):
        x = np.concatenate([grid, _action_channels(action)], axis=1)
        self.h = self.r2.forward(self.c2.forward(
            self.r1.forward(self.c1.forward(x))))
        next_grid = self.grid_head.forward(self.h)
        dlogit = self.done_head.forward(self.h).mean(axis=(2, 3))[:, 0]
        return next_grid, _sigmoid(dlogit)

    def _modules(self):
        return [self.c1, self.c2, self.grid_head, self.done_head]

    def zero_grad(self):
        for m in self._modules():
            m.zero_grad()

    def clip_grads(self, max_norm=5.0):
        for m in self._modules():
            m.clip_grads(max_norm)

    def adam_step(self):
        for m in self._modules():
            m.adam_step()

    def save(self, path):
        out = {}
        for i, m in enumerate(self._modules()):
            out[f"w{i}"] = m.W; out[f"b{i}"] = m.b
        np.savez(path, **out)

    def load(self, path):
        d = np.load(path)
        for i, m in enumerate(self._modules()):
            m.W = d[f"w{i}"]; m.b = d[f"b{i}"]

    def update(self, grid, action, next_grid, done):
        n = len(grid)
        self.zero_grad()
        pred_grid, pred_done = self.forward(grid, action)

        # next grid MSE（逐通道，连续/0/1 统一）
        g_grid = (pred_grid - next_grid) / n
        g_h = self.grid_head.backward(g_grid)

        # done BCE（全局平均，死亡加权）
        target = np.asarray(done, DTYPE).reshape(-1)
        w = np.where(target > .5, 20.0, 1.0)
        d_done = (pred_done - target) * w / n
        # 反向：全局平均的梯度均匀回灌到 10×10
        g_dlogit = np.zeros((n, 1, 10, 10), DTYPE)
        g_dlogit[:, 0] = (d_done * pred_done * (1 - pred_done)).reshape(n, 1, 1) / 100
        g_h += self.done_head.backward(g_dlogit)

        g_h = self.r2.backward(g_h); g_h = self.c2.backward(g_h)
        g_h = self.r1.backward(g_h); self.c1.backward(g_h)

        self.clip_grads(5.0)
        self.adam_step()
        return {
            "grid_mse": float(np.mean((pred_grid - next_grid) ** 2)),
            "done_acc": float(np.mean((pred_done > .5) == target)),
        }
