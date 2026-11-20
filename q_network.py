"""Q 网络：ReLU MLP，最后一层线性输出各动作的 Q 值，内置 Adam 优化。

与 nn.Net 的区别：nn.Net 末端用 sigmoid（为 GA 的 0/1 输出设计）；
Q 学习需要输出任意实数的期望回报，因此末端不做激活，并支持按目标值做
Huber/MSE 反向更新。
"""
import numpy as np

DTYPE = np.float32


class QNetwork:
    def __init__(self, sizes, lr=0.001, huber_delta=1.0):
        self.sizes = sizes
        self.lr = lr
        self.huber_delta = huber_delta
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            # He 初始化，适配 ReLU
            self.W.append(np.random.randn(sizes[i + 1], sizes[i]).astype(DTYPE)
                          * np.sqrt(2.0 / sizes[i]))
            self.b.append(np.zeros(sizes[i + 1], dtype=DTYPE))
        # Adam 一阶/二阶矩
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        self.acts = [np.asarray(X, dtype=DTYPE)]
        self.zs = []
        a = self.acts[0]
        for i in range(len(self.W)):
            z = a @ self.W[i].T + self.b[i]
            self.zs.append(z)
            a = np.maximum(0, z) if i < len(self.W) - 1 else z  # 末层线性
            self.acts.append(a)
        return self.acts[-1]

    def update(self, X, target):
        """按目标 Q 向量计算 Huber/MSE 损失并 Adam 更新一步。"""
        self.forward(X)
        err = self.acts[-1] - np.asarray(target, dtype=DTYPE)
        n = len(X)
        if self.huber_delta and self.huber_delta > 0:
            # Huber：小误差平方、大误差线性，抑制异常目标的冲击
            d = np.where(np.abs(err) <= self.huber_delta, err,
                         self.huber_delta * np.sign(err)) / n
        else:
            d = err / n

        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.append(d.T @ self.acts[i])
            gb.append(d.sum(0))
            if i > 0:
                d = (d @ self.W[i]) * (self.zs[i - 1] > 0)
        gW.reverse()
        gb.reverse()

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * gW[i] ** 2
            self.W[i] -= self.lr * (self.mW[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.vW[i] / (1 - b2 ** self.t)) + eps)
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb[i] ** 2
            self.b[i] -= self.lr * (self.mb[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.vb[i] / (1 - b2 ** self.t)) + eps)
