import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
import numpy as np
from latent_world_model import LatentWorldModel

np.random.seed(1)
wm = LatentWorldModel(state_dim=6, n_actions=2, z_dim=4, lr=1e-3)
o = np.random.randn(5, 6).astype(np.float32)
o2 = np.random.randn(5, 6).astype(np.float32)
a = np.array([0, 1, 0, 1, 0])
r = np.array([0.1, -0.02, 0.1, 10.0, -0.02], np.float32)


zt_fixed = wm.E.forward(o2).copy()       # stop-gradient：目标端固定


def total_loss():
    z = wm.E.forward(o)
    x = np.concatenate([z, wm._onehot(a)], 1)
    zp = wm.D.forward(x)
    rp = wm.R.forward(zp)
    return (np.sum((zp - zt_fixed) ** 2) + np.sum((rp[:, 0] - r) ** 2)) / len(o)


# 解析梯度（只跑 backward，不更新）
n = len(o)
z = wm.E.forward(o)
zt = zt_fixed
x = np.concatenate([z, wm._onehot(a)], 1)
zp = wm.D.forward(x)
rp = wm.R.forward(zp)
gR = wm.R.backward(2 * (rp - r.reshape(-1, 1)) / n)
gX = wm.D.backward(2 * (zp - zt) / n + gR)
wm.E.backward(gX[:, :4])

h = 1e-5
for name, net in [("E", wm.E), ("D", wm.D), ("R", wm.R)]:
    worst_rel, worst_abs = 0.0, 0.0
    for li in range(len(net.W)):
        for _ in range(3):
            r0, c0 = np.random.randint(net.W[li].shape[0]), np.random.randint(net.W[li].shape[1])
            orig = net.W[li][r0, c0]
            net.W[li][r0, c0] = orig + h; lp = total_loss()
            net.W[li][r0, c0] = orig - h; lm = total_loss()
            net.W[li][r0, c0] = orig
            num = (lp - lm) / (2 * h)
            ana = net.gW[li][r0, c0]
            worst_rel = max(worst_rel, abs(num - ana) / max(1e-8, abs(num)))
            worst_abs = max(worst_abs, abs(num - ana))
    print(f"{name} 相对误差={worst_rel:.2e} 绝对误差={worst_abs:.2e}")
