import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

found = None
for seed in range(1, 60):
    env = SnakeEnv(); st = env.reset(seed)
    heads, acts = [], []
    for t in range(80):
        a = ag.act(st, False)
        heads.append(env.body[0]); acts.append(a)
        st, _, done, _ = env.step(a)
        if done: break
    # 检测：后 40 步只经过 4 个不同格子 → 封闭小循环
    if not done and len(set(heads[-40:])) <= 4:
        found = (seed, heads, acts); break

if found:
    seed, heads, acts = found
    print("命中 seed =", seed, "循环格子:", sorted(set(heads[-40:])))
    print("一个周期动作:", acts[-8:])
    print("对应坐标:", heads[-8:])
else:
    print("前59个seed未发现四格循环")
