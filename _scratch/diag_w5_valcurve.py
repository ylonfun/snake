"""奖励头泛化曲线：train/val 拆分，看留出 MSE 随 epoch 的变化与最佳点。"""
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
s = env.reset(3000)
while len(G2) < 6000:
    a = behavior.act(s, training=False)
    s, r, done, _ = env.step(a)
    G2.append(s); R.append(r)
    if done:
        s = env.reset(3000 + len(G2))
G2 = np.asarray(G2, np.float32); R = np.asarray(R, np.float32)
z = model.encode(G2)
tr, va = np.arange(4000), np.arange(4000, 6000)

net = MLP([64, 64, 1], lr=1e-3)
for step in range(1, 6001):
    idx = tr[np.random.randint(0, 4000, 64)]
    net.zero_grad()
    g = (net.forward(z[idx]) - R[idx].reshape(-1, 1)) / 64
    net.backward(g); net.adam_step()
    if step % 500 == 0:
        ptr = net.forward(z[tr])[:, 0]; pva = net.forward(z[va])[:, 0]
        print("步%d 训练MSE=%.3f 验证MSE=%.3f 验证吃苹果=%.2f" % (
            step, np.mean((ptr-R[tr])**2), np.mean((pva-R[va])**2),
            pva[R[va] > 5].mean()))
