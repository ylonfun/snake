import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel

model = LatentGridModel(z_dim=64)
env = SnakeEnv(state_mode="grid")

# 收集一小批真实 transition
grids, acts, rews, nxts, dones = [], [], [], [], []
s = env.reset(0)
for _ in range(64):
    a = np.random.randint(4)
    s2, r, done, _ = env.step(a)
    grids.append(s); acts.append(a); rews.append(r); nxts.append(s2); dones.append(done)
    s = env.reset() if done else s2

g = np.asarray(grids, np.float32)
g2 = np.asarray(nxts, np.float32)
z = model.encode(g)
print("encode 输出形状:", z.shape, "范围:", round(z.min(), 2), round(z.max(), 2))
pz2 = model.step_latent(z, acts)
print("一步想象形状:", pz2.shape)
print("reward:", model.reward_of(z).shape, "cont:", model.cont_prob(z).shape)
for i in range(5):
    info = model.update(g, acts, rews, g2, dones)
print("5 次 update 无报错，损失:", {k: round(v, 4) for k, v in info.items()})
