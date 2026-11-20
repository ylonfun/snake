"""诊断 15×15 空转：蛇头到食物的 BFS 距离与动作方向。"""
import os, sys
from collections import deque
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

_MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))


def bfs_dist(head, food, body_set, rows, cols):
    q, seen = deque([(head, 0)]), {head}
    while q:
        (x, y), d = q.popleft()
        if (x, y) == food:
            return d
        for dx, dy in _MOVES:
            p = (x+dx, y+dy)
            if 0 <= p[0] < cols and 0 <= p[1] < rows and p not in body_set and p not in seen:
                seen.add(p); q.append((p, d+1))
    return -1


agent = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
env = SnakeEnv(rows=15, cols=15, state_mode="grid", deadline_growth=1.0)
s = env.reset(5000)
for step in range(400):
    body_set = set(env.body[1:])
    d = bfs_dist(env.body[0], env.food, body_set, 15, 15)
    if step % 20 == 0:
        print(f"step{step} 头={env.body[0]} 食物={env.food} BFS距离={d} "
              f"体长={len(env.body)}")
    a = agent.act(s, training=False, env=env)
    s, _, done, info = env.step(a)
    if done:
        print("结束", info)
        break
