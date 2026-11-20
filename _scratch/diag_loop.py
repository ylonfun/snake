import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

env = SnakeEnv()
st = env.reset(1)
heads, acts = [], []
for t in range(120):
    a = ag.act(st, False)
    heads.append(env.body[0]); acts.append(a)
    st, _, done, info = env.step(a)
    if done:
        print("结束于", t, info); break
print("身体长度(恒定):", len(env.body))
print("前20步动作:", acts[:20])
print("前20步头坐标:", heads[:20])
uniq = sorted(set(heads))
print("经过的不同格子数:", len(uniq), "格子:", uniq)
