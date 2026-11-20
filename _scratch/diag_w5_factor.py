"""因素分解：学习率 / batch size / 裁剪 对奖励头拟合的影响。"""
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


def train(batch, lr, clip, steps=3000):
    net = MLP([64, 64, 1], lr=lr)
    for _ in range(steps):
        idx = (np.arange(N) if batch >= N else np.random.randint(0, N, batch))
        net.zero_grad()
        g = (net.forward(z[idx]) - R[idx].reshape(-1, 1)) / len(idx)
        net.backward(g)
        if clip:
            net.clip_grads(5.0)
        net.adam_step()
    p = net.forward(z)[:, 0]
    return p[food].mean(), p[~food].mean()


conds = [
    ("在线复现 b64 lr3e-4 clip", 64, 3e-4, True),
    ("b64 lr1e-3 clip", 64, 1e-3, True),
    ("b64 lr1e-3 无clip", 64, 1e-3, False),
    ("全量 lr1e-3 无clip", N, 1e-3, False),
    ("全量 lr3e-4 无clip", N, 3e-4, False),
]
for tag, b, lr, c in conds:
    f, n = train(b, lr, c)
    print(f"{tag:28s} 吃苹果={f:.2f} 普通={n:.3f}")
