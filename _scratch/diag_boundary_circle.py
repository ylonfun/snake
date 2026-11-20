import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")


def on_edge(p):
    return p[0] in (0, 9) or p[1] in (0, 9)


def show(env):
    body = set(env.body)
    print("身体长度", len(env.body), "食物", env.food)
    for y in range(10):
        row = []
        for x in range(10):
            if (x, y) == env.body[0]:
                row.append("H")
            elif (x, y) in body:
                row.append("o")
            elif (x, y) == env.food:
                row.append("F")
            else:
                row.append(".")
        print("  " + " ".join(row))
    print("气味场(保留2位小数):")
    for y in range(10):
        print("  " + " ".join(f"{env.scent[y,x]:.2f}" for x in range(10)))


for seed in range(5000, 5010):
    env = SnakeEnv(); st = env.reset(seed)
    edge_run = 0
    while True:
        a = ag.act(st, False)
        st2, _, done, info = env.step(a)
        if on_edge(env.body[0]):
            edge_run += 1
        else:
            edge_run = 0
        if edge_run == 12:   # 连续12步贴边 = 正在绕边界
            print(f"=== seed {seed} 连续{edge_run}步沿边界，动作{a} ===")
            show(env)
            print("各动作mask (True安全):", ag._legal_mask(st))
            break
        st = st2
        if done:
            break
    if edge_run == 12:
        break
