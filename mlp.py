"""通用 MLP 积木：前向 + 反向（暴露对输入的梯度）+ Adam，用于搭建隐空间世界模型。

与 q_network 的区别：这里把"反向"和"参数更新"拆开，外部可手动串联多个 MLP
（编码器→动力学→奖励头），并在需要处用 stop-gradient。
- 末层可选 tanh（隐状态限幅）或 sigmoid（终止概率）；
- backward 梯度累加，配合 zero_grad 支持同一网络对多个时刻/样本反复 backward。
"""
import numpy as np

DTYPE = np.float32


class MLP:
    def __init__(self, sizes, lr=1e-3, out_act="linear"):
        self.sizes = list(sizes)
        self.lr = lr
        self.out_act = out_act
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            self.W.append(np.random.randn(sizes[i + 1], sizes[i]).astype(DTYPE)
                          * np.sqrt(2.0 / sizes[i]))
            self.b.append(np.zeros(sizes[i + 1], dtype=DTYPE))
        self.gW = [np.zeros_like(w) for w in self.W]
        self.gb = [np.zeros_like(x) for x in self.b]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(x) for x in self.b]
        self.vb = [np.zeros_like(x) for x in self.b]
        self.t = 0

    def zero_grad(self):
        for i in range(len(self.W)):
            self.gW[i].fill(0)
            self.gb[i].fill(0)

    def forward(self, X):
        self.acts = [np.asarray(X, DTYPE)]
        self.zs = []
        a = self.acts[0]
        for i in range(len(self.W)):
            z = a @ self.W[i].T + self.b[i]
            self.zs.append(z)
            if i < len(self.W) - 1:
                a = np.maximum(0, z)                   # 隐藏层 ReLU
            elif self.out_act == "tanh":
                a = np.tanh(z)
            elif self.out_act == "sigmoid":
                zc = np.clip(z, -60, 60)
                a = np.where(zc >= 0, 1.0 / (1.0 + np.exp(-zc)),
                             np.exp(zc) / (1.0 + np.exp(zc)))
            else:
                a = z
            self.acts.append(a)
        return self.acts[-1]

    def backward(self, g):
        """累加各层参数梯度，返回对输入的梯度。"""
        d = np.asarray(g, DTYPE)
        for i in reversed(range(len(self.W))):
            if i == len(self.W) - 1 and self.out_act in ("tanh", "sigmoid"):
                a = self.acts[i + 1]
                d = d * (1 - a ** 2) if self.out_act == "tanh" else d * a * (1 - a)
            elif i < len(self.W) - 1:                  # 隐藏层过 ReLU 导数
                d = d * (self.zs[i] > 0)
            self.gW[i] += d.T @ self.acts[i]
            self.gb[i] += d.sum(0)
            d = d @ self.W[i]
        return d

    def clip_grads(self, max_norm):
        total = sum(np.sum(g ** 2) for g in self.gW)
        total += sum(np.sum(g ** 2) for g in self.gb)
        norm = np.sqrt(total)
        if norm > max_norm:
            scale = max_norm / norm
            for i in range(len(self.W)):
                self.gW[i] *= scale
                self.gb[i] *= scale

    def adam_step(self):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * self.gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * self.gW[i] ** 2
            self.W[i] -= self.lr * (self.mW[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.vW[i] / (1 - b2 ** self.t)) + eps)
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * self.gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * self.gb[i] ** 2
            self.b[i] -= self.lr * (self.mb[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.vb[i] / (1 - b2 ** self.t)) + eps)
