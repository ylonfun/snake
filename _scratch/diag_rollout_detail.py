import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

STAGE = 4
agent = DreamerAgent(freeze_world=True, horizon=5)
agent.load_world("world_frozen.npz")
agent.load(os.path.join("dreamer_ckpt", f"stage_{STAGE}.npz"))
w = agent.world
env = SnakeEnv()
s = env.reset(7000)
z = w.encode(np.asarray([s], np.float32))
print("初始食物", env.food, "身体", env.body)

print("\n=== 各动作一步想象 ===")
for a in range(4):
    cur = w.step(z, np.asarray([a]))
    r_raw = w.R.forward(cur)[0, 0]
    r_clip = w.reward(cur)[0]
    c = w.C.forward(cur)[0, 0]
    print(f"动作{a}: R原始={r_raw:7.2f} R裁剪={r_clip:7.2f} C继续={c:6.3f}")

print("\n=== 沿 actor 贪心 rollout 5 步 ===")
cur = z
for t in range(5):
    a = int(agent.actor.forward(cur).argmax(1)[0])
    cur = w.step(cur, np.asarray([a]))
    r_raw = w.R.forward(cur)[0, 0]
    r_clip = w.reward(cur)[0]
    c = w.C.forward(cur)[0, 0]
    print(f"步{t+1} 动作{a}: R原始={r_raw:7.2f} R裁剪={r_clip:7.2f} C继续={c:6.3f}")
