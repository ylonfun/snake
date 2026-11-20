import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

# 找一个能吃到至少2个苹果的局
target = None
for seed in range(5000, 5040):
    env = SnakeEnv(); st = env.reset(seed)
    log, eat_steps = [], []
    for t in range(300):
        a = ag.act(st, False)
        food_rays = [d for d in range(8) if st[8 + d * 4 + 1] > 0.5]
        log.append((t, a, env.body[0], env.steps_since_food,
                    env.visit_count[env.body[0][1], env.body[0][0]], food_rays))
        st, r, done, info = env.step(a)
        if r > 5: eat_steps.append(t)
        if done: break
    if len(eat_steps) >= 2:
        target = (eat_steps[0], log); break

first_eat, log = target
print("第一次吃苹果发生在第", first_eat, "步，打印其前后片段：")
print("步 | 动作 | 头坐标 | 饥饿计数 | 本格访问数 | 看见苹果的射线")
for row in log:
    t = row[0]
    if first_eat - 3 <= t <= first_eat + 10:
        mark = "  <== 吃到苹果" if t == first_eat else ""
        print(f"{t:2d} | {row[1]} | {row[2]} | {row[3]:2d} | {row[4]} | {row[5]}{mark}")
