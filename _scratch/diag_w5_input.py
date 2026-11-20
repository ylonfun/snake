"""验证奖励头输入：只用 z2 vs (z,action) vs (z,z2) 对吃苹果的可分性。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from mlp import MLP
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
model = LatentGridModel(); model.load(os.path.join("cnn_ckpt", "world_cnn.npz"))
env = SnakeEnv(state_mode="grid")
G, A, R, G2 = [], [], [], []
s = env.reset(3000)
while len(G) < 5000:
    a = behavior.act(s, training=False)
    s2, r, done, _ = env.step(a)
    G.append(s); A.append(a); R.append(r); G2.append(s2)
    s = env.reset(3000 + len(G)) if done else s2
G = np.asarray(G, np.float32); G2 = np.asarray(G2, np.float32)
R = np.asarray(R, np.float32); A = np.asarray(A)
z, z2 = model.encode(G), model.encode(G2)
y = (R > 5).astype(np.float32)

# 动作 one-hot
ah = np.zeros((len(A), 4), np.float32); ah[np.arange(len(A)), A] = 1


def probe(X, steps=1500, lr=1e-2):
    net = MLP([X.shape[1], 1], lr=lr)
    for _ in range(steps):
        net.zero_grad()
        g = (net.forward(X) - y.reshape(-1, 1)) / len(X)
        net.backward(g); net.adam_step()
    return np.corrcoef(net.forward(X)[:, 0], y)[0, 1]


print("线性探针吃苹果相关性:")
print("  只用 z2        :", round(probe(z2), 3))
print("  (z, action)    :", round(probe(np.concatenate([z, ah], 1)), 3))
print("  (z, z2)        :", round(probe(np.concatenate([z, z2], 1)), 3))
