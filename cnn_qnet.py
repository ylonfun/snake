"""CNN Q 网络：完整多通道网格 -> 卷积提空间特征 -> 各动作 Q 值。

结构：Conv(6→16)+ReLU → Pool → Conv(16→32)+ReLU → Pool → Flatten → MLP(128→64→4)。
用 Sequential 风格逐层前向、逆序反向，聚合可学参数统一 Adam/裁剪。
"""
import numpy as np

from conv import Conv2D, ReLU, MaxPool2D, Flatten
from mlp import MLP

DTYPE = np.float32


class CNNQNet:
    def __init__(self, in_channels=6, lr=1e-3, huber_delta=1.0):
        self.huber_delta = huber_delta
        self.layers = [
            Conv2D(in_channels, 16, 3, pad=1, lr=lr), ReLU(),
            MaxPool2D(2),
            Conv2D(16, 32, 3, pad=1, lr=lr), ReLU(),
            MaxPool2D(2),
            Flatten(),
            MLP([128, 64, 4], lr),
        ]

    def forward(self, X):
        h = np.asarray(X, DTYPE)
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def _learnable(self):
        return [l for l in self.layers if hasattr(l, "adam_step")
                and l.__class__.__name__ in ("Conv2D", "MLP")]

    def zero_grad(self):
        for l in self._learnable():
            l.zero_grad()

    def clip_grads(self, max_norm):
        for l in self._learnable():
            l.clip_grads(max_norm)

    def adam_step(self):
        for l in self._learnable():
            l.adam_step()

    def update(self, X, target):
        pred = self.forward(X)
        err = pred - np.asarray(target, DTYPE)
        n = len(X)
        if self.huber_delta and self.huber_delta > 0:
            d = np.where(np.abs(err) <= self.huber_delta, err,
                         self.huber_delta * np.sign(err)) / n
        else:
            d = err / n
        self.zero_grad()
        self.backward(d)
        self.clip_grads(5.0)
        self.adam_step()
