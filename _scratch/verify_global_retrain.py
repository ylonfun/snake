import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from collections import Counter
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")

causes = Counter(); scores = []; bodies = []; steps = []; edge_after = []
for seed in range(5000, 5040):
    env = SnakeEnv(); st = env.reset(seed)
    after = []
    while True:
        a = ag.act(st, False)
        st2, r, done, info = env.step(a)
        if r == env.r_food:
            after = []
        else:
            hx, hy = env.body[0]
            after.append(hx in (0, 9) or hy in (0, 9))
        if done:
            break
        st = st2
    causes[env.death_cause] += 1
    scores.append(info["score"]); bodies.append(len(env.body)); steps.append(info["steps"])
    edge_after.append(np.mean(after) if after else 0)

print("40局 均分", round(np.mean(scores), 1), " 峰值", max(scores))
print("平均体长", round(np.mean(bodies), 1), " 平均步数", round(np.mean(steps), 0))
print("死因分布", dict(causes))
print("最后一次吃果后在边界的平均占比", round(np.mean(edge_after), 2))
