"""泛化诊断：两批独立数据的 z 分布差异 + 奖励头在各自数据上的表现。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
model = LatentGridModel(); model.load(os.path.join("cnn_ckpt", "world_cnn.npz"))


def collect(seed0, n=2000):
    G2, R = [], []
    e = SnakeEnv(state_mode="grid")
    s = e.reset(seed0)
    while len(G2) < n:
        a = behavior.act(s, training=False)
        s, r, done, _ = e.step(a)
        G2.append(s); R.append(r)
        if done:
            s = e.reset(seed0 + len(G2))
    return np.asarray(G2, np.float32), np.asarray(R, np.float32)


gA, rA = collect(1000)
gB, rB = collect(9000)
zA, zB = model.encode(gA), model.encode(gB)
print("zA 均值范围:", round(zA.mean(0).min(), 3), round(zA.mean(0).max(), 3),
      "zB:", round(zB.mean(0).min(), 3), round(zB.mean(0).max(), 3))
print("两批 z 均值差异 最大:", round(np.abs(zA.mean(0)-zB.mean(0)).max(), 3),
      "整体均值:", round(np.abs(zA.mean(0)-zB.mean(0)).mean(), 3))

for tag, z, r in [("A", zA, rA), ("B", zB, rB)]:
    p = model.reward_of(z)
    food = r > 5
    print("数据%s 吃苹果预测=%.2f(n=%d) 普通=%.3f" % (
        tag, p[food].mean(), food.sum(), p[~food].mean()))
