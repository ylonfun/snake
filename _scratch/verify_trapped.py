import os, sys

root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv, DIRECTIONS
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")

for seed in (5011, 5033):
    env = SnakeEnv(); st = env.reset(seed)
    while True:
        st2, _, done, _ = env.step(ag.act(st, False))
        if done:
            break
        st = st2
    if env.death_cause != "wall":
        continue
    # 用环境自身规则，在“最后一步之前”的状态重放：需要重新跑到最后一步前
    env = SnakeEnv(); st = env.reset(seed)
    hist = []
    while True:
        a = ag.act(st, False)
        hist.append((env.body.copy(), env.direction))
        st2, _, done, _ = env.step(a)
        if done:
            break
        st = st2
    body, direction = hist[-1]
    hard = set(body[:-1])
    print(f"seed {seed} 头{body[0]} 当前方向{direction}")
    n_safe = 0
    for act0 in range(4):
        nd = DIRECTIONS[act0]
        if nd == (-direction[0], -direction[1]):
            nd = direction
        h = (body[0][0] + nd[0], body[0][1] + nd[1])
        wall = not (0 <= h[0] < 10 and 0 <= h[1] < 10)
        self_hit = h in hard
        ok = not (wall or self_hit)
        n_safe += ok
        print(f"  动作{act0} -> {h} 墙={wall} 身={self_hit} 安全={ok}")
    print("  安全动作数 =", n_safe)
