import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from settings import DIRECTIONS

for L in (10, 40, 80, 90):
    env = SnakeEnv(init_len=L, min_len=L)
    env.reset(1)
    hx, hy = env.body[0]
    safe = 0
    for dx, dy in DIRECTIONS:
        nx, ny = hx + dx, hy + dy
        in_grid = 0 <= nx < 10 and 0 <= ny < 10
        if in_grid and (nx, ny) not in env.body[:-1]:
            safe += 1
    empty = 100 - len(env.body)
    print(f"身体{L}节：剩余空格{empty}个，蛇头当前安全动作{safe}个")
