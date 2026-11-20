"""验证小 batch 长训：b64 跑足够多 epoch 能否追上全量。"""
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

net = MLP([64, 64, 1], lr=1e-3)
for step in range(1, 50001):
    idx = np.random.randint(0, N, 64)
    net.zero_grad()
    g = (net.forward(z[idx]) - R[idx].reshape(-1, 1)) / 64
    net.backward(g); net.adam_step()
    if step in (3000, 10000, 25000, 50000):
        p = net.forward(z)[:, 0]
        epochs = step * 64 // N
        print(f"步{step}(约{epochs} epoch) 吃苹果={p[food].mean():.2f} "
              f"普通={p[~food].mean():.3f}")
