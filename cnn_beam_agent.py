"""Beam search 束搜索规划器：环境克隆 + flood fill + 可达尾巴的多路径启发式搜索。

每个首动作扩展后，每步保留 top-B 个累计折扣分支，枚举全部动作而非沿 Q 贪心，
因此能跳出 Q 网络的平台。评分含三项：环境奖励、flood fill 连通空间项、以及
tail-reachability（蛇头能否到达当前尾巴）的拓扑生存惩罚——后者能识别"连通格数
不少但形状无法容纳长蛇"的口袋，是突破困死的关键。
尺寸与归一化均从环境读取，支持任意棋盘大小。不依赖 Q 网络做决策，属离线强规划。
"""
from collections import deque

import numpy as np

from cnn_dqn_agent import CNNDQNAgent
from cnn_safe_agent import flood_fill

_MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))


def can_reach_tail(head, tail, obstacles, rows=10, cols=10):
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
        for dx, dy in _MOVES:
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if 0 <= nx < cols and 0 <= ny < rows and p not in obs and p not in seen:
                q.append(p)
    return False


def bfs_to_food(head, food, obstacles, rows, cols):
    """蛇头到食物的最短可行距离（避开身体），不可达返回大值。"""
    if head == food:
        return 0
    seen, q = set(), deque([(head, 0)])
    while q:
        (x, y), d = q.popleft()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        for dx, dy in _MOVES:
            p = (x + dx, y + dy)
            if p in seen or p in obstacles:
                continue
            if not (0 <= p[0] < cols and 0 <= p[1] < rows):
                continue
            if p == food:
                return d + 1
            q.append((p, d + 1))
    return rows * cols


class CNNBeamAgent(CNNDQNAgent):
    def __init__(self, depth=10, beam=3, space_weight=8.0, tail_pen=5.0,
                 food_weight=0.0, **kw):
        super().__init__(**kw)
        self.depth, self.beam = depth, beam
        self.space_weight, self.tail_pen = space_weight, tail_pen
        self.food_weight = food_weight
        self._scratch = None

    def _node_score(self, sc, rr):
        rows, cols = sc.rows, sc.cols
        total_cells = rows * cols
        body_set = set(sc.body[1:])
        sp = flood_fill(sc.body[0], body_set, rows, cols)
        tail_ok = can_reach_tail(sc.body[0], sc.body[-1], body_set, rows, cols)
        pen = 0.0 if tail_ok else self.tail_pen
        fd = bfs_to_food(sc.body[0], sc.food, body_set, rows, cols)
        goal = self.food_weight * (1.0 - fd / total_cells)
        return rr + self.space_weight * sp / total_cells - pen + goal

    def _beam_value(self, env, first_action):
        if self._scratch is None:
            from rl_env import SnakeEnv
            self._scratch = SnakeEnv(
                rows=env.rows, cols=env.cols, state_mode="grid",
                deadline_growth=getattr(env, "deadline_growth", 0.0),
                food_deadline=env.food_deadline)
        sc = self._scratch
        sc.restore_state(env.clone_state())
        _, r, done, _ = sc.step(first_action)
        head = self._node_score(sc, r)
        branches = [] if done else [{"score": head, "snap": sc.clone_state()}]
        finished = [head] if done else []
        discount = self.gamma
        for _ in range(self.depth - 1):
            candidates = []
            for b in branches:
                sc.restore_state(b["snap"])
                for a in range(self.n_actions):
                    snap = sc.clone_state()
                    _, rr, dn, _ = sc.step(a)
                    candidates.append({"score": b["score"] + discount * (
                        self._node_score(sc, rr)),
                        "snap": sc.clone_state(), "done": dn})
                    sc.restore_state(snap)
            finished += [c["score"] for c in candidates if c["done"]]
            keep = sorted((c for c in candidates if not c["done"]),
                          key=lambda x: -x["score"])
            branches = keep[:self.beam]
            discount *= self.gamma
        return max(finished + [b["score"] for b in branches])

    def act(self, state, training=True, env=None):
        if training or env is None:
            return super().act(state, training=training)
        best_a, best_v = 0, -np.inf
        for a in range(self.n_actions):
            v = self._beam_value(env, a)
            if v > best_v:
                best_v, best_a = v, a
        return best_a
