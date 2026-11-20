import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from q_network import QNetwork

rng = np.random.RandomState(0)
net = QNetwork([5, 4, 3], lr=0.01, huber_delta=0)   # 纯 MSE 便于检查
X = rng.randn(8, 5).astype(np.float32)
target = rng.randn(8, 3).astype(np.float32)


def mse_loss():
    out = net.forward(X)
    return ((out - target) ** 2).sum() / (2 * len(X))


# 解析梯度（手动反向）
net.forward(X)
d = (net.acts[-1] - target) / len(X)
ana = {}
for i in reversed(range(len(net.W))):
    ana[i] = (d.T @ net.acts[i]).copy()
    if i > 0:
        d = (d @ net.W[i]) * (net.zs[i - 1] > 0)

eps = 1e-4
worst = 0
for li in range(2):
    for idx in np.ndindex(net.W[li].shape):
        orig = net.W[li][idx]
        net.W[li][idx] = orig + eps
        lp = mse_loss()
        net.W[li][idx] = orig - eps
        lm = mse_loss()
        net.W[li][idx] = orig
        num = (lp - lm) / (2 * eps)
        rel = abs(num - ana[li][idx]) / max(1e-8, abs(num) + abs(ana[li][idx]))
        worst = max(worst, rel)

print(f"Q网络梯度检查最大相对误差={worst:.2e}（需 <1e-5）")
assert worst < 1e-5
print("M2 梯度检查通过")
