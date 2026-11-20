import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

agent = DreamerAgent(freeze_world=True, horizon=5)
agent.load_world("world_frozen.npz")
agent.load(os.path.join("dreamer_ckpt", "final.npz"))
env = SnakeEnv()

scores, lens, steps, causes, loops = [], [], [], {}, 0
for seed in range(5000, 5040):
    s = env.reset(seed)
    seen = set()
    while True:
        key = (tuple(env.body), env.food)
        if key in seen:
            loops += 1
        seen.add(key)
        a = agent.act(s, training=False)
        s, r, done, info = env.step(a)
        if done:
            break
    scores.append(info["score"])
    lens.append(len(env.body))
    steps.append(info["steps"])
    causes[env.death_cause] = causes.get(env.death_cause, 0) + 1

print("同口径40局 (seed 5000-5039, final)")
print(f"均分={np.mean(scores):.2f} 峰值={max(scores)}")
print(f"平均体长={np.mean(lens):.1f} 平均步数={np.mean(steps):.0f}")
print(f"出现状态重复环的局数={loops}")
print(f"死因分布={causes}")
