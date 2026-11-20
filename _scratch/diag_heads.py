import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from dreamer_agent import DreamerAgent
from train_rl import load_checkpoint

# 收集一批真实数据
beh = DQNAgent(state_mode="local"); load_checkpoint(beh, "dqn_ckpt/final.npz"); beh.eps = 0.3
S, A, R, S2, D = [], [], [], [], []
e = SnakeEnv(); st = e.reset()
for _ in range(8000):
    a = beh.act(st, True)
    s2, r, done, _ = e.step(a)
    S.append(st); A.append(a); R.append(r); S2.append(s2); D.append(done)
    st = s2
    if done: st = e.reset()
S = np.array(S, np.float32); A = np.array(A); R = np.array(R, np.float32); D = np.array(D)

ag = DreamerAgent(freeze_world=True); ag.load_world("world_frozen.npz")
z2 = ag.world.step(ag.world.encode(S), A)
cpred = ag.world.C.forward(z2)[:, 0]
rpred = ag.world.R.forward(z2)[:, 0]

print("真实 done 比例:", round(D.mean(), 4))
# 终止头校准：对真实 done=1 的样本，C 预测的 continue 概率
fatal_idx = D == 1
print("真实死亡样本数:", fatal_idx.sum())
print("死亡样本上 C 预测 continue 均值:", round(cpred[fatal_idx].mean(), 3),
      "其中被误判继续(>0.5)比例:", round((cpred[fatal_idx] > 0.5).mean(), 3))
print("全部样本 C 预测 continue 均值:", round(cpred.mean(), 3))
# 奖励头
print("奖励头预测 MSE:", round(((rpred-R)**2).mean(), 3))
