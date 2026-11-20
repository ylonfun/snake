"""Beam search 配置扫描：分数 vs 耗时（每配置 10 局）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from _scratch.exp_beam import beam_value


def run(depth, beam, n=10):
    scratch = SnakeEnv(state_mode="grid")
    scores, t0 = [], time.time()
    for seed in range(5000, 5000 + n):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            best_a, best_v = 0, -np.inf
            for a in range(4):
                v = beam_value(env, a, scratch, depth, beam)
                if v > best_v:
                    best_v, best_a = v, a
            s, _, done, info = env.step(best_a)
            if done:
                scores.append(info["score"]); break
    return np.mean(scores), np.max(scores), (time.time()-t0)/n


for depth, beam in ((6, 2), (8, 2), (6, 3), (10, 3)):
    m, mx, sec = run(depth, beam)
    print(f"depth{depth:2d} beam{beam} 均分={m:.1f} 峰值={mx} 每局={sec:.1f}s")
