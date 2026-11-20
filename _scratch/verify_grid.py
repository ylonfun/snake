import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv

env = SnakeEnv(state_mode="grid")
seen = set()
for seed in range(20000):
    g = env.reset(seed)
    assert g.shape == (6, 10, 10), g.shape
    seen.add(tuple(g.reshape(-1)))
print("grid 形状:", g.shape, "通道数", g.shape[0])
print("20000 个局面去重后不同 grid 数 =", len(seen), "(局部观测仅 52)")

# 通道正确性：头/身/食物通道的非零位置
g0 = env.reset(0)
head_pos = tuple(np.argwhere(g0[0] == 1)[0][::-1])
food_pos = tuple(np.argwhere(g0[2] == 1)[0][::-1])
print("头通道位置(yx->xy):", head_pos, "实际头", env.body[0])
print("食物通道位置:", food_pos, "实际食物", env.food)
print("气味通道范围:", round(g0[3].min(), 5), "-", round(g0[3].max(), 5))
print("边界通道非零格数:", int(g0[5].sum()))

# 闭环跑 300 步无报错
env.reset(1)
for _ in range(300):
    a = np.random.randint(4)
    g, r, done, info = env.step(a)
    if done:
        env.reset()
print("闭环 300 步无报错")
