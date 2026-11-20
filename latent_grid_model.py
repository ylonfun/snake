"""W5 潜在世界模型（Dreamer-lite，针对完整可观测 grid MDP）。

grid 完整可观测（非 POMDP），简化 RSSM：
- 在线编码器：CNN 把 (6,10,10) 压成 tanh 限幅 z；
- 解码器：MLP 从 z 重建完整 grid，强制 z 保留全部信息（修复表征坍缩：仅动力学
  损失时编码器会丢弃稀疏的吃苹果信息，导致奖励头不可学）；
- 目标编码器：在线编码器的 EMA 副本，为动力学提供稳定目标（修复"移动靶"）；
- 动力学：MLP(z + 动作 one-hot) -> 下一 z；奖励头 / 继续头从 z 预测。
"""
import numpy as np

from conv import Conv2D, ReLU, MaxPool2D, Flatten
from mlp import MLP

DTYPE = np.float32
GRID_CELLS = 6 * 10 * 10


def _one_hot(a, n=4):
    out = np.zeros((len(a), n), DTYPE)
    out[np.arange(len(a)), np.asarray(a).reshape(-1)] = 1.0
    return out


class LatentGridModel:
    def __init__(self, z_dim=64, lr=3e-4, enc_lr=1e-4, ema_tau=0.005):
        self.z_dim = z_dim
        self.ema_tau = ema_tau

        # 在线编码器
        self.e1 = Conv2D(6, 16, 3, pad=1, lr=enc_lr)
        self.r1 = ReLU(); self.p1 = MaxPool2D(2)
        self.e2 = Conv2D(16, 32, 3, pad=1, lr=enc_lr)
        self.r2 = ReLU(); self.p2 = MaxPool2D(2)
        self.flat = Flatten()
        self.enc = MLP([128, z_dim], enc_lr, out_act="tanh")

        # 目标编码器（独立实例，只 EMA 不反传）
        self.te1 = Conv2D(6, 16, 3, pad=1, lr=enc_lr)
        self.tr1 = ReLU(); self.tp1 = MaxPool2D(2)
        self.te2 = Conv2D(16, 32, 3, pad=1, lr=enc_lr)
        self.tr2 = ReLU(); self.tp2 = MaxPool2D(2)
        self.tflat = Flatten()
        self.tenc = MLP([128, z_dim], enc_lr, out_act="tanh")
        self._copy_target()

        # 解码器 / 动力学 / 奖励 / 继续
        self.dec = MLP([z_dim, 256, GRID_CELLS], lr)
        self.dyn = MLP([z_dim + 4, 128, z_dim], lr, out_act="tanh")
        self.rew = MLP([z_dim, 64, 1], lr)
        self.cont = MLP([z_dim, 32, 1], lr, out_act="sigmoid")

    # ---- 前向 ----
    def _run_encoder(self, grid, c1, act1, pool1, c2, act2, pool2, flatten, mlp):
        h = c1.forward(grid)
        h = act1.forward(h); h = pool1.forward(h)
        h = c2.forward(h)
        h = act2.forward(h); h = pool2.forward(h)
        h = flatten.forward(h)
        return mlp.forward(h)

    def encode(self, grid):
        return self._run_encoder(grid, self.e1, self.r1, self.p1, self.e2,
                                 self.r2, self.p2, self.flat, self.enc)

    def encode_target(self, grid):
        return self._run_encoder(grid, self.te1, self.tr1, self.tp1, self.te2,
                                 self.tr2, self.tp2, self.tflat, self.tenc)

    def encode_backward(self, gz):
        g = self.enc.backward(gz)
        g = self.flat.backward(g)
        g = self.p2.backward(g); g = self.r2.backward(g); g = self.e2.backward(g)
        g = self.p1.backward(g); g = self.r1.backward(g)
        return self.e1.backward(g)

    def reconstruct(self, z):
        return self.dec.forward(z).reshape(-1, 6, 10, 10)

    def step_latent(self, z, action):
        return self.dyn.forward(np.concatenate([z, _one_hot(action)], axis=1))

    def reward_of(self, z):
        return self.rew.forward(z)[:, 0]

    def cont_prob(self, z):
        return self.cont.forward(z)[:, 0]

    # ---- 目标网络 ----
    def _copy_target(self):
        self.te1.W = self.e1.W.copy(); self.te1.b = self.e1.b.copy()
        self.te2.W = self.e2.W.copy(); self.te2.b = self.e2.b.copy()
        for i in range(len(self.enc.W)):
            self.tenc.W[i] = self.enc.W[i].copy()
            self.tenc.b[i] = self.enc.b[i].copy()

    def ema_update(self):
        t = self.ema_tau
        self.te1.W = (1 - t) * self.te1.W + t * self.e1.W
        self.te1.b = (1 - t) * self.te1.b + t * self.e1.b
        self.te2.W = (1 - t) * self.te2.W + t * self.e2.W
        self.te2.b = (1 - t) * self.te2.b + t * self.e2.b
        for i in range(len(self.enc.W)):
            self.tenc.W[i] = (1 - t) * self.tenc.W[i] + t * self.enc.W[i]
            self.tenc.b[i] = (1 - t) * self.tenc.b[i] + t * self.enc.b[i]

    # ---- 存档 ----
    def save(self, path):
        out = {}
        pairs = [("e1", self.e1), ("e2", self.e2), ("enc", self.enc),
                 ("dec", self.dec), ("dyn", self.dyn),
                 ("rew", self.rew), ("cont", self.cont)]
        for name, m in pairs:
            if isinstance(m.W, list):
                for i in range(len(m.W)):
                    out[f"{name}_W{i}"] = m.W[i]; out[f"{name}_b{i}"] = m.b[i]
            else:
                out[f"{name}_W"] = m.W; out[f"{name}_b"] = m.b
        np.savez(path, **out)

    def load(self, path):
        d = np.load(path)
        pairs = [("e1", self.e1), ("e2", self.e2), ("enc", self.enc),
                 ("dec", self.dec), ("dyn", self.dyn),
                 ("rew", self.rew), ("cont", self.cont)]
        for name, m in pairs:
            if isinstance(m.W, list):
                for i in range(len(m.W)):
                    m.W[i] = d[f"{name}_W{i}"]; m.b[i] = d[f"{name}_b{i}"]
            else:
                m.W = d[f"{name}_W"]; m.b = d[f"{name}_b"]
        self._copy_target()

    # ---- 参数管理 ----
    def _modules(self):
        return [self.e1, self.e2, self.enc, self.dec,
                self.dyn, self.rew, self.cont]

    def zero_grad(self):
        for m in self._modules():
            m.zero_grad()

    def clip_grads(self, max_norm=5.0):
        for m in self._modules():
            m.clip_grads(max_norm)

    def adam_step(self):
        for m in self._modules():
            m.adam_step()

    def update(self, grid, action, reward, next_grid, done):
        n = len(grid)
        self.zero_grad()

        z = self.encode(grid)
        z2 = self.encode(next_grid)
        tz2 = self.encode_target(next_grid)

        # 0) 重建损失（强制信息保留，梯度传回编码器）
        recon = self.reconstruct(z)
        d_dec = (recon - grid).reshape(n, -1) / n
        g_dec = self.dec.backward(d_dec)
        self.encode_backward(g_dec)

        # 1) 动力学：目标来自 EMA 目标编码器
        pred_z2 = self.step_latent(z, action)
        d_dyn = (pred_z2 - tz2.copy()) / n
        g_dyn = self.dyn.backward(d_dyn)
        self.encode_backward(g_dyn[:, :self.z_dim])

        # 2) 奖励
        pred_r = self.rew.forward(z2)
        d_rew = ((pred_r - np.asarray(reward, DTYPE).reshape(-1, 1)) / n)
        self.encode_backward(self.rew.backward(d_rew))

        # 3) 继续（死亡加权）
        target_c = (1.0 - np.asarray(done, DTYPE)).reshape(-1, 1)
        pc = self.cont.forward(z2)
        weight = np.where(target_c > 0.5, 1.0, 30.0)
        d_cont = (pc - target_c) * weight / n
        self.encode_backward(self.cont.backward(d_cont))

        self.clip_grads(5.0)
        self.adam_step()
        self.ema_update()
        return {
            "rec": float(np.mean((recon - grid) ** 2)),
            "dyn": float(np.mean((pred_z2 - tz2) ** 2)),
            "rew": float(np.mean((pred_r[:, 0] - np.asarray(reward)) ** 2)),
        }
