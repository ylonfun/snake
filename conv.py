"""纯 numpy 卷积积木：Conv2D / MaxPool2D / Flatten，配合 mlp.MLP 搭建 CNN。

卷积用 im2col 转成矩阵乘（前向），反向用 col2im 累加输入梯度，梯度推导与
全连接一致、便于数值检查。各层接口对齐 mlp.MLP：forward/backward/zero_grad/
clip_grads/adam_step；无可学参数的层给出空参数列表。
张量约定：(N, C, H, W)。
"""
import numpy as np

DTYPE = np.float32


def im2col(x, k, stride, pad):
    n, c, h, w = x.shape
    out_h = (h + 2 * pad - k) // stride + 1
    out_w = (w + 2 * pad - k) // stride + 1
    xp = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)])
    col = np.zeros((n, c, k, k, out_h, out_w), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            col[:, :, i, j, :, :] = xp[:, :,
                                       i:i + stride * out_h:stride,
                                       j:j + stride * out_w:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return col, out_h, out_w


def col2im(col, shape, k, stride, pad, out_h, out_w):
    n, c, h, w = shape
    col = col.reshape(n, out_h, out_w, c, k, k).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=col.dtype)
    for i in range(k):
        for j in range(k):
            xp[:, :, i:i + stride * out_h:stride,
                  j:j + stride * out_w:stride] += col[:, :, i, j, :, :]
    if pad:
        return xp[:, :, pad:-pad, pad:-pad]
    return xp


class Conv2D:
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=0, lr=1e-3):
        self.k, self.stride, self.pad = ksize, stride, pad
        self.out_ch = out_ch
        self.W = (np.random.randn(out_ch, in_ch, ksize, ksize).astype(DTYPE)
                  * np.sqrt(2.0 / (in_ch * ksize * ksize)))
        self.b = np.zeros(out_ch, dtype=DTYPE)
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.lr, self.t = lr, 0

    def forward(self, X):
        self.x_shape = X.shape
        self.col, self.oh, self.ow = im2col(X, self.k, self.stride, self.pad)
        self.wcol = self.W.reshape(self.out_ch, -1)
        out = self.col @ self.wcol.T + self.b
        out = out.reshape(X.shape[0], self.oh, self.ow, self.out_ch)
        return out.transpose(0, 3, 1, 2)

    def backward(self, dout):
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_ch)
        self.gb += dout_r.sum(0)
        self.gW += (dout_r.T @ self.col).reshape(self.W.shape)
        dcol = dout_r @ self.wcol
        return col2im(dcol, self.x_shape, self.k, self.stride, self.pad,
                      self.oh, self.ow)

    def zero_grad(self):
        self.gW.fill(0); self.gb.fill(0)

    def _params(self):
        return [("W", self.W, self.gW, self.mW, self.vW),
                ("b", self.b, self.gb, self.mb, self.vb)]

    def clip_grads(self, max_norm):
        norm = np.sqrt(np.sum(self.gW ** 2) + np.sum(self.gb ** 2))
        if norm > max_norm:
            s = max_norm / norm
            self.gW *= s; self.gb *= s

    def adam_step(self):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, w, g, m, v in self._params():
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g ** 2
            mh = m / (1 - b1 ** self.t)
            vh = v / (1 - b2 ** self.t)
            w -= self.lr * mh / (np.sqrt(vh) + eps)


class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dout):
        return dout * self.mask

    def zero_grad(self):
        pass

    def clip_grads(self, n):
        pass

    def adam_step(self):
        pass


class MaxPool2D:
    def __init__(self, ksize=2, stride=None):
        self.k = ksize
        self.stride = stride or ksize

    def forward(self, X):
        n, c, h, w = X.shape
        self.oh = (h - self.k) // self.stride + 1
        self.ow = (w - self.k) // self.stride + 1
        col, _, _ = im2col(X, self.k, self.stride, 0)
        col = col.reshape(-1, self.k * self.k)
        self.arg = col.argmax(1)
        out = col.max(1)
        out = out.reshape(n, self.oh, self.ow, c).transpose(0, 3, 1, 2)
        self.x_shape = X.shape
        self.n, self.c = n, c
        return out

    def backward(self, dout):
        dcol = np.zeros((dout.size, self.k * self.k), dtype=dout.dtype)
        dflat = dout.transpose(0, 2, 3, 1).reshape(-1)
        dcol[np.arange(len(self.arg)), self.arg] = dflat
        dcol = dcol.reshape(self.n * self.oh * self.ow, -1)
        return col2im(dcol, self.x_shape, self.k, self.stride, 0,
                      self.oh, self.ow)

    def zero_grad(self):
        pass

    def clip_grads(self, n):
        pass

    def adam_step(self):
        pass


class Flatten:
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

    def zero_grad(self):
        pass

    def clip_grads(self, n):
        pass

    def adam_step(self):
        pass
