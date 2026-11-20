"""W6 诊断：死亡局上 flood fill 连通空间的轨迹，验证可提前识别困死。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_lookahead_agent import CNNLookaheadAgent
from settings import DIRECTIONS


def flood_fill(head, body_set):
    seen, stack = set(), [head]
    while stack:
        x, y = stack.pop()
        if (x, y) in seen:
            continue
        if x < 0 or x >= 10 or y < 0 or y >= 10 or (x, y) in body_set:
            continue
        seen.add((x, y))
        for dx, dy in DIRECTIONS:
            stack.append((x + dx, y + dy))
    return len(seen)


def action_spaces(env):
    """每个安全动作执行后的连通空间大小。"""
    out = {}
    for a in range(4):
        snap = env.clone_state()
        _, _, done, _ = env.step(a)
        if not done:
            out[a] = flood_fill(env.body[0], set(env.body[1:]))
        env.restore_state(snap)
    return out


agent = CNNLookaheadAgent(lookahead=10)
agent.load(os.path.join("cnn_ckpt", "best.npz"))
for seed in [5000, 5005]:
    env = SnakeEnv(state_mode="grid")
    s = env.reset(seed)
    trace = []
    while True:
        a = agent.act(s, training=False, env=env)
        spaces = action_spaces(env)
        chosen = spaces.get(a, 0)
        trace.append((chosen, max(spaces.values()) if spaces else 0))
        s, _, done, info = env.step(a)
        if done:
            break
    print("=" * 50, "seed", seed, "分", info["score"], "步数", len(trace))
    print("  最后10步 (选择动作空间, 最大可用空间):")
    for c, m in trace[-10:]:
        print("   ", c, m)
