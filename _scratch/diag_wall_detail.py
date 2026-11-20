import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv, DIRECTIONS
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

# 找一个撞墙局
for seed in range(5000, 5020):
    env = SnakeEnv(); st = env.reset(seed)
    log = []
    while True:
        a = ag.act(st, False)
        # 统计当前四个方向的安全动作数
        safe = 0
        for aa in range(4):
            nd = DIRECTIONS[aa]
            nx, ny = env.body[0][0] + nd[0], env.body[0][1] + nd[1]
            if 0 <= nx < env.cols and 0 <= ny < env.rows and (nx, ny) not in env.body[:-1]:
                safe += 1
        log.append((env.body[0], a, safe, st[8:40:4].round(2)))
        st, _, done, _ = env.step(a)
        if done:
            if env.death_cause == "wall":
                print("seed", seed, "撞墙，死前6步：")
                print("头坐标 | 动作(0上1下2左3右) | 安全动作数 | 8方向(1/距离)")
                for row in log[-6:]:
                    print(row)
            break
    if env.death_cause == "wall":
        break
