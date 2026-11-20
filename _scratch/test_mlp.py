import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
import numpy as np
from mlp import MLP

np.random.seed(0)
net = MLP([8, 6, 4, 3])
X = np.random.randn(5, 8).astype(np.float32)
y = np.random.randn(5, 3).astype(np.float32)


def loss():
    out = net.forward(X)
    return np.sum((out - y) ** 2) / len(X)


out = net.forward(X)
g_out = 2 * (out - y) / len(X)
g_in = net.backward(g_out)

# 1) 参数梯度：抽查每层若干权重
h = 1e-5
worst = 0.0
for li in range(len(net.W)):
    for _ in range(4):
        r, c = np.random.randint(net.W[li].shape[0]), np.random.randint(net.W[li].shape[1])
        orig = net.W[li][r, c]
        net.W[li][r, c] = orig + h
        lp = loss()
        net.W[li][r, c] = orig - h
        lm = loss()
        net.W[li][r, c] = orig
        num = (lp - lm) / (2 * h)
        ana = net.gW[li][r, c]
        worst = max(worst, abs(num - ana) / max(1e-8, abs(num)))
print("参数梯度最大相对误差:", f"{worst:.2e}")

# 2) 输入梯度：抽查
worst_in = 0.0
for r, c in [(0, 0), (2, 7), (4, 3)]:
    orig = X[r, c]
    X[r, c] = orig + h
    lp = loss()
    X[r, c] = orig - h
    lm = loss()
    X[r, c] = orig
    num = (lp - lm) / (2 * h)
    ana = g_in[r, c]
    worst_in = max(worst_in, abs(num - ana) / max(1e-8, abs(num)))
print("输入梯度最大相对误差:", f"{worst_in:.2e}")
