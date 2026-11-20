import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
import conv as C
C.DTYPE = np.float64   # 梯度检查用 float64 提高精度


def err(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))


# ---- Conv2D: stride=1, pad=1, k=3 ----
np.random.seed(0)
X = np.random.randn(2, 3, 5, 5)
layer = C.Conv2D(3, 4, 3, stride=1, pad=1)
dout = np.random.randn(2, 4, 5, 5)
out = layer.forward(X)
dX = layer.backward(dout)
gW, gb = layer.gW.copy(), layer.gb.copy()

fn = lambda: layer.forward(X)
nX = np.zeros_like(X)
eps = 1e-6
it = np.nditer(X, flags=["multi_index"])
while not it.finished:
    i = it.multi_index; old = X[i]
    X[i] = old + eps; lp = (fn() * dout).sum()
    X[i] = old - eps; lm = (fn() * dout).sum()
    X[i] = old
    nX[i] = (lp - lm) / (2 * eps); it.iternext()
print("Conv2D 输入梯度相对误差:", err(nX, dX))

# 权重数值梯度（抽样前若干元素，控制耗时）
nW = np.zeros_like(layer.W)
it = np.nditer(layer.W, flags=["multi_index"])
cnt = 0
while not it.finished and cnt < 40:
    i = it.multi_index; old = layer.W[i]
    layer.W[i] = old + eps; lp = (fn() * dout).sum()
    layer.W[i] = old - eps; lm = (fn() * dout).sum()
    layer.W[i] = old
    nW[i] = (lp - lm) / (2 * eps); it.iternext(); cnt += 1
m = nW != 0
print("Conv2D 权重梯度相对误差:", err(nW[m], gW[m]))

nb = np.zeros_like(layer.b)
for i in range(len(layer.b)):
    old = layer.b[i]
    layer.b[i] = old + eps; lp = (fn() * dout).sum()
    layer.b[i] = old - eps; lm = (fn() * dout).sum()
    layer.b[i] = old
    nb[i] = (lp - lm) / (2 * eps)
print("Conv2D 偏置梯度相对误差:", err(nb, gb))

# ---- MaxPool2D ----
X2 = np.random.randn(2, 3, 4, 4)
pool = C.MaxPool2D(2)
dout2 = np.random.randn(2, 3, 2, 2)
o2 = pool.forward(X2)
dX2 = pool.backward(dout2)
nX2 = np.zeros_like(X2)
fn2 = lambda: pool.forward(X2)
it = np.nditer(X2, flags=["multi_index"])
while not it.finished:
    i = it.multi_index; old = X2[i]
    X2[i] = old + eps; lp = (fn2() * dout2).sum()
    X2[i] = old - eps; lm = (fn2() * dout2).sum()
    X2[i] = old
    nX2[i] = (lp - lm) / (2 * eps); it.iternext()
print("MaxPool 输入梯度相对误差:", err(nX2, dX2))
print("梯度检查完成")
