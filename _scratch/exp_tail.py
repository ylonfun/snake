"""Tail-reachability 增强 beam search 实验。"""
import os, sys, time
from collections import deque
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_safe_agent import flood_fill


def can_reach_tail(head, tail, obstacles):
    if head == tail:
        return True
    obs = obstacles - {tail}
    seen, q = set(), deque([head])
    while q:
        x, y = q.popleft()
        if (x, y) == tail:
            return True
        if (x, y) in seen:
            continue
        seen.add((x, y))
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if 0 <= nx < 10 and 0 <= ny < 10 and p not in obs and p not in seen:
                q.append(p)
    return False


def beam_value_tail(env, first_action, scratch, depth=10, beam=3, sw=8.0,
                    gamma=0.95, tail_pen=5.0):
    scratch.restore_state(env.clone_state())
    _, r, done, _ = scratch.step(first_action)
    body_set = set(scratch.body[1:])
    sp = flood_fill(scratch.body[0], body_set)
    tail_ok = can_reach_tail(scratch.body[0], scratch.body[-1], body_set)
    pen = 0 if tail_ok else tail_pen
    head_score = r + sw * sp / 100 - pen
    branches = [] if done else [{"score": head_score, "snap": scratch.clone_state()}]
    finished = [head_score] if done else []
    discount = gamma
    for _ in range(depth - 1):
        candidates = []
        for b in branches:
            scratch.restore_state(b["snap"])
            for a in range(4):
                snap = scratch.clone_state()
                _, rr, dn, _ = scratch.step(a)
                bs = set(scratch.body[1:])
                spp = flood_fill(scratch.body[0], bs)
                ok = can_reach_tail(scratch.body[0], scratch.body[-1], bs)
                pn = 0 if ok else tail_pen
                candidates.append({"score": b["score"] + discount * (
                    rr + sw * spp / 100 - pn),
                    "snap": scratch.clone_state(), "done": dn})
                scratch.restore_state(snap)
        finished += [c["score"] for c in candidates if c["done"]]
        keep = sorted((c for c in candidates if not c["done"]),
                      key=lambda x: -x["score"])
        branches = keep[:beam]
        discount *= gamma
    return max(finished + [b["score"] for b in branches])


def run(n=10):
    scratch = SnakeEnv(state_mode="grid")
    scores, t0 = [], time.time()
    for seed in range(5000, 5000 + n):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            best_a, best_v = 0, -np.inf
            for a in range(4):
                v = beam_value_tail(env, a, scratch)
                if v > best_v:
                    best_v, best_a = v, a
            s, _, done, info = env.step(best_a)
            if done:
                scores.append(info["score"]); break
    return np.mean(scores), np.max(scores), (time.time()-t0)/n


m, mx, sec = run()
print(f"beam+tail 均分={m:.1f} 峰值={mx} 每局={sec:.1f}s")
