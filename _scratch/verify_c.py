import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from dreamer_agent import DreamerAgent
from train_rl import load_checkpoint

beh = DQNAgent(state_mode="local"); load_checkpoint(beh, "dqn_ckpt/final.npz"); beh.eps = 0.3
S, A, D = [], [], []
e = SnakeEnv(); st = e.reset()
for _ in range(8000):
    a = beh.act(st, True)
    s2, r, done, _ = e.step(a)
    S.append(st); A.append(a); D.append(done); st = s2
    if done: st = e.reset()
S = np.array(S, np.float32); A = np.array(A); D = np.array(D)
ag = DreamerAgent(freeze_world=True); ag.load_world("world_frozen.npz")
cp = ag.world.C.forward(ag.world.step(ag.world.encode(S), A))[:, 0]
fatal = D == 1
print("死亡样本上 continue 均值:", round(cp[fatal].mean(), 3),
      "误判继续比例:", round((cp[fatal] > 0.5).mean(), 3))
print("正常样本 continue 均值:", round(cp[~fatal].mean(), 3))
