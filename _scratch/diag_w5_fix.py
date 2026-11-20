"""验证修复：对比原始奖励 vs 归一化奖励在在线 mini-batch + clip 下的拟合。"""
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
G2, R = [], []
s = env.reset()
while len(G2) < 3000:
    a = behavior.act(s, training=False)
    s2, r, done, _ = env.step(a)
    G2.append(s2); R.append(r)
    s = env.reset() if done else s2
G2 = np.asarray(G2, np.float32); R = np.asarray(R, np.float32)
z = model.encode(G2); food = R > 5; N = len(R)

for tag, scale in [("原始奖励", 1.0), ("归一化(/10)", 0.1)]:
    net = MLP([64, 64, 1], lr=3e-4)
    for _ in range(3000):
        idx = np.random.randint(0, N, 64)
        net.zero_grad()
        g = (net.forward(z[idx]) - (R[idx] * scale).reshape(-1, 1)) / 64
        net.backward(g); net.clip_grads(5.0); net.adam_step()
    p = net.forward(z)[:, 0] / scale   # 还原到原始尺度
    print(f"{tag}: 吃苹果预测={p[food].mean():.2f} 普通={p[~food].mean():.3f}")
