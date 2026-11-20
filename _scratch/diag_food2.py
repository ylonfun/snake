import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from dreamer_agent import DreamerAgent
from train_rl import load_checkpoint

beh = DQNAgent(state_mode="local"); load_checkpoint(beh, "dqn_ckpt/final.npz"); beh.eps = 0.3
S, A, R = [], [], []
e = SnakeEnv(); st = e.reset()
for _ in range(12000):
    a = beh.act(st, True)
    s2, r, done, _ = e.step(a)
    S.append(st); A.append(a); R.append(r); st = s2
    if done: st = e.reset()
S = np.array(S, np.float32); A = np.array(A); R = np.array(R)
ag = DreamerAgent(freeze_world=True); ag.load_world("world_frozen.npz")
rp = ag.world.R.forward(ag.world.step(ag.world.encode(S), A))[:, 0]

food = R > 5
print("吃食物样本数:", food.sum(), "占比:", round(food.mean(), 4))
print("真实吃食物时 R 头预测均值:", round(rp[food].mean(), 2), "(真实+10)")
print("普通步 R 头预测均值:", round(rp[~food].mean(), 3), "(真实-0.02)")
