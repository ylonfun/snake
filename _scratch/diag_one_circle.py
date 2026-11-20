import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")

# 从身体1节开始，记录前40步完整轨迹，检测位置是否重复
for seed in range(0, 12):
    env = SnakeEnv(); st = env.reset(seed)
    traj = []
    for _ in range(40):
        a = ag.act(st, False)
        st2, r, done, info = env.step(a)
        hx, hy = env.body[0]
        traj.append((env.body[0], a, round(float(env.scent[hy, hx]), 4),
                     round(float(r), 3), len(env.body)))
        if done:
            break
        st = st2
    pos = [t[0] for t in traj]
    has_repeat = len(set(pos)) < len(pos)
    if has_repeat and len(traj) >= 6:
        print(f"seed{seed} 初始体长1，前{len(traj)}步出现位置重复，苹果{env.food}")
        for t in traj:
            print("  位置", t[0], "动作", t[1], "气味", t[2], "奖励", t[3], "体长", t[4])
        break
