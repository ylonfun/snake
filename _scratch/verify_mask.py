import os, sys

root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

rows = []
for seed in range(5000, 5040):
    env = SnakeEnv(); st = env.reset(seed)
    while True:
        st, _, done, info = env.step(ag.act(st, False))
        if done:
            rows.append((env.death_cause, len(env.body), info["score"], info["steps"]))
            break

cause = np.array([r[0] for r in rows])
length = np.array([r[1] for r in rows])
foods = np.array([r[2] for r in rows])
steps = np.array([r[3] for r in rows])
print("死亡类型计数:", {c: int((cause == c).sum()) for c in set(cause)})
print(f"死亡体长 均值{length.mean():.1f} 范围{length.min()}-{length.max()}")
print(f"吃苹果数 均值{foods.mean():.1f} 范围{foods.min()}-{foods.max()}")
print(f"步数 均值{steps.mean():.0f} 范围{steps.min()}-{steps.max()}")
