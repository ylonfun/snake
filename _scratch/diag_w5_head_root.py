"""排查头部欠拟合根因：(1) MLP/Adam 单元拟合能力 (2) z 对吃苹果的可分性。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from mlp import MLP
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

# (1) MLP 最小拟合测试
np.random.seed(0)
X = np.random.randn(256, 8).astype(np.float32)
tw = np.random.randn(8, 1).astype(np.float32)
Y = (X @ tw)[:, 0]
net = MLP([8, 32, 1], lr=1e-2)
for _ in range(1500):
    net.zero_grad()
    g = (net.forward(X) - Y.reshape(-1, 1)) / len(X)
    net.backward(g); net.adam_step()
print("(1) MLP 拟合相关性:", round(np.corrcoef(net.forward(X)[:, 0], Y)[0, 1], 3))

# (2) z 对吃苹果的可分性
behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
model = LatentGridModel(); model.train_encoder = False
env = SnakeEnv(state_mode="grid")
G2, R = [], []
s = env.reset()
while len(G2) < 3000:
    a = behavior.act(s, training=False)
    s2, r, done, _ = env.step(a)
    G2.append(s2); R.append(r)
    s = env.reset() if done else s2
G2 = np.asarray(G2, np.float32); R = np.asarray(R)
z = model.encode(G2)
food = R > 5
print("(2) 吃苹果 n=", food.sum(), "普通 n=", (~food).sum())
print("  z 各维度 吃苹果组均值 abs 最大:",
      round(np.abs(z[food].mean(0) - z[~food].mean(0)).max(), 3))
# 线性探针：用 z 线性回归预测"是否吃苹果"
probe = MLP([z.shape[1], 1], lr=1e-2)
y = food.astype(np.float32)
for _ in range(800):
    probe.zero_grad()
    p = probe.forward(z)
    g = (p - y.reshape(-1, 1)) / len(z)
    probe.backward(g); probe.adam_step()
p = probe.forward(z)[:, 0]
print("  线性探针预测吃苹果相关性:", round(np.corrcoef(p, y)[0, 1], 3))
