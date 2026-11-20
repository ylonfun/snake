"""CNN-DQN + 内部模型想象前瞻（W5 重设计）：用学到的 GridForwardModel 替代环境克隆。

继承 CNNDQNAgent，仅在评估（贪心）时对每个候选动作在"想象 grid"上 rollout K 步：
模型预测 next grid/done，确定性规则给奖励，后续沿 Q 贪心，累计折扣后选最优。
训练时保持 ε-greedy 不变。与 W4 的唯一区别是把真实环境克隆换成学出来的前向模型。
"""
import os
import numpy as np

from cnn_dqn_agent import CNNDQNAgent
from grid_forward_model import GridForwardModel


class CNNImaginedAgent(CNNDQNAgent):
    def __init__(self, lookahead=5, world_path=None, **kw):
        super().__init__(**kw)
        self.lookahead = lookahead
        self.world = GridForwardModel()
        if world_path is None:
            world_path = os.path.join("cnn_ckpt", "grid_forward.npz")
        self.world.load(world_path)

    @staticmethod
    def _reward(grid, next_grid, done):
        if done:
            return -10.0
        shape = grid.shape[1:]
        fy, fx = np.unravel_index(np.argmax(grid[2]), shape)
        hy, hx = np.unravel_index(np.argmax(next_grid[0]), shape)
        if (hy, hx) == (fy, fx):
            return 10.0
        return -0.02 + float(next_grid[3, hy, hx])

    def _imagined_value(self, grid, first_action):
        total, discount = 0.0, 1.0
        cur = grid[np.newaxis]
        ng, pd = self.world.forward(cur, [first_action])
        done = bool(pd[0] > .5)
        total += discount * self._reward(grid, ng[0], done)
        discount *= self.gamma
        cur = ng
        for _ in range(self.lookahead - 1):
            if done:
                break
            a = int(self.q.forward(cur).argmax(1)[0])
            ng, pd = self.world.forward(cur, [a])
            done = bool(pd[0] > .5)
            total += discount * self._reward(cur[0], ng[0], done)
            discount *= self.gamma
            cur = ng
        return total

    def act(self, state, training=True, env=None):
        if training or env is None:
            return super().act(state, training=training)
        best_a, best_v = 0, -np.inf
        for a in range(self.n_actions):
            v = self._imagined_value(state, a)
            if v > best_v:
                best_v, best_a = v, a
        return best_a
