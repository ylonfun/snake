import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
print("运行时 R_W1 均值 =", round(float(ag.world.R.W[1].mean()), 5),
      "（npz文件里应为 0.08342）")
print("r_lo/r_hi =", ag.world.r_lo, ag.world.r_hi, "food_r_weight =", ag.world.food_r_weight)

# 直接用新 R 对一批真实状态预测，按真实奖励分组
from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint
beh = DQNAgent(state_mode="local"); load_checkpoint(beh, "dqn_ckpt/final.npz"); beh.eps=0.3
env = SnakeEnv(); st = env.reset()
groups = {}
for _ in range(3000):
    a = beh.act(st[:40], True)
    z = ag.world.encode(np.asarray([st], np.float32))
    rp = float(ag.world.R.forward(z)[0, 0])
    st2, r, done, _ = env.step(a)
    groups.setdefault("food" if r == 10 else "normal", []).append(rp)
    if done:
        st = env.reset()
    else:
        st = st2
print("新R 普通步预测均值 =", round(np.mean(groups["normal"]), 3))
print("新R 吃苹果步预测均值 =", round(np.mean(groups["food"]), 3))
