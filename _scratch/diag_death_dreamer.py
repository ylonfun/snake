import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv, DIRECTIONS
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

N = 40
rows = []
for seed in range(5000, 5000 + N):
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

print("=== 死亡类型 ===")
for c in ["wall", "self", "hunger", "full"]:
    m = cause == c
    if m.sum():
        print(f"{c:7s} {m.sum():2d}局  占比{100*m.mean():.0f}%  该组平均体长{length[m].mean():.1f}")

print("\n=== 死亡时体长（节）===")
print(f"均值{length.mean():.1f} 中位数{np.median(length):.0f} 最小{length.min()} 最大{length.max()}")
print("体长分布:", dict(zip(*np.unique(length, return_counts=True))))

print("\n=== 死亡时吃苹果数 ===")
print(f"均值{foods.mean():.1f} 中位数{np.median(foods):.0f} 最小{foods.min()} 最大{foods.max()}")

print("\n=== 死亡时步数 ===")
print(f"均值{steps.mean():.0f} 中位数{np.median(steps):.0f} 最小{steps.min()} 最大{steps.max()}")
