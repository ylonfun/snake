"""W6 安全空间规划器：W4 前瞻价值 + flood fill 连通空间项。

继承 CNNLookaheadAgent，决策时对每个候选动作同时计算多步前瞻价值与执行后的连通
空格数，按"前瞻价值 + 空间权重×归一化空间"组合选择，避免走进连通空间骤降的口袋。
"""
import numpy as np

from cnn_lookahead_agent import CNNLookaheadAgent
from settings import DIRECTIONS


def flood_fill(head, body_set, rows=10, cols=10):
    seen, stack = set(), [head]
    while stack:
        x, y = stack.pop()
        if (x, y) in seen:
            continue
        if x < 0 or x >= cols or y < 0 or y >= rows or (x, y) in body_set:
            continue
        seen.add((x, y))
        for dx, dy in DIRECTIONS:
            stack.append((x + dx, y + dy))
    return len(seen)


class CNNSafeAgent(CNNLookaheadAgent):
    def __init__(self, lookahead=10, space_weight=12.0, **kw):
        super().__init__(lookahead=lookahead, **kw)
        self.space_weight = space_weight

    def _flood_space(self, env, action):
        snap = env.clone_state()
        _, _, done, _ = env.step(action)
        space = -1 if done else flood_fill(
            env.body[0], set(env.body[1:]), env.rows, env.cols)
        env.restore_state(snap)
        return space

    def act(self, state, training=True, env=None):
        if training or env is None:
            return super().act(state, training=training)
        cells = env.rows * env.cols
        best_a, best_v = 0, -np.inf
        for a in range(self.n_actions):
            lookahead_v = self._lookahead_value(env, a)
            space = self._flood_space(env, a)
            combo = lookahead_v if space < 0 else (
                lookahead_v + self.space_weight * space / cells)
            if combo > best_v:
                best_v, best_a = combo, a
        return best_a


class CNNMultiSpaceAgent(CNNSafeAgent):
    """多步空间规划：rollout 的每一步都累计 flood fill，而非只看首动作。

    捕捉"连通空间在未来如何演化"，能识别首步空间相同、但一个通向开阔、一个通向
    死胡同的差异，是当前最强的规划器。
    """
    def __init__(self, lookahead=10, space_weight=8.0, **kw):
        super().__init__(lookahead=lookahead, space_weight=space_weight, **kw)

    def _multi_value(self, env, first_action):
        snap = env.clone_state()
        cells = env.rows * env.cols
        total, discount = 0.0, 1.0
        s, r, done, _ = env.step(first_action)
        space = flood_fill(env.body[0], set(env.body[1:]), env.rows, env.cols)
        total += discount * (r + self.space_weight * space / cells)
        discount *= self.gamma
        for _ in range(self.lookahead - 1):
            if done:
                break
            a = int(self.q.forward(np.asarray([s], np.float32)).argmax(1)[0])
            s, r, done, _ = env.step(a)
            space = flood_fill(env.body[0], set(env.body[1:]), env.rows, env.cols)
            total += discount * (r + self.space_weight * space / cells)
            discount *= self.gamma
        env.restore_state(snap)
        return total

    def act(self, state, training=True, env=None):
        if training or env is None:
            return super(CNNSafeAgent, self).act(state, training=training)
        best_a, best_v = 0, -np.inf
        for a in range(self.n_actions):
            v = self._multi_value(env, a)
            if v > best_v:
                best_v, best_a = v, a
        return best_a

