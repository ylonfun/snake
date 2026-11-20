"""Beam search 束搜索规划实验（环境克隆 + flood fill）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_safe_agent import flood_fill


def beam_value(env, first_action, scratch, depth=8, beam=3, sw=8.0, gamma=0.95):
    scratch.restore_state(env.clone_state())
    _, r, done, _ = scratch.step(first_action)
    sp = flood_fill(scratch.body[0], set(scratch.body[1:]))
    branches = [{"score": r + sw * sp / 100, "snap": scratch.clone_state()}]
    finished = [{"score": r + sw * sp / 100}] if done else []
    if done:
        branches = []
    discount = gamma
    for _ in range(depth - 1):
        candidates = []
        for b in branches:
            scratch.restore_state(b["snap"])
            for a in range(4):
                snap = scratch.clone_state()
                _, rr, dn, _ = scratch.step(a)
                spp = flood_fill(scratch.body[0], set(scratch.body[1:]))
                candidates.append({"score": b["score"] + discount * (
                    rr + sw * spp / 100), "snap": scratch.clone_state(),
                    "done": dn})
                scratch.restore_state(snap)
        keep = [c for c in candidates if not c["done"]]
        finished += [c for c in candidates if c["done"]]
        keep.sort(key=lambda x: -x["score"])
        branches = keep[:beam]
        discount *= gamma
    pool = finished + branches
    return max(c["score"] for c in pool)


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


m, mx, sec = run(8, 3)
print(f"beam(depth8 beam3) 均分={m:.1f} 峰值={mx} 每局耗时={sec:.1f}s")
