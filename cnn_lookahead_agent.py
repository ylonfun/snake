"""CNN-DQN + 有限步前瞻（W4 验证）：用真实环境克隆做"想象"，零模型误差。

继承 CNNDQNAgent，仅在评估（贪心）时对每个候选动作在环境副本里 rollout K 步，
用真实环境计算多步折扣回报，选综合最优而非单步贪心；训练时保持 ε-greedy 不变。
这是 Dyna 规划的理想化特例（模型=真实环境），先验证"多步前瞻"逻辑本身；
W5 再把环境克隆替换为学出来的潜在动力学模型。
"""
import numpy as np

from cnn_dqn_agent import CNNDQNAgent


class CNNLookaheadAgent(CNNDQNAgent):
    def __init__(self, lookahead=5, **kw):
        super().__init__(**kw)
        self.lookahead = lookahead

    def _lookahead_value(self, env, first_action):
        snap = env.clone_state()
        total, discount = 0.0, 1.0
        s, r, done, _ = env.step(first_action)
        total += discount * r
        discount *= self.gamma
        for _ in range(self.lookahead - 1):
            if done:
                break
            a = int(self.q.forward(np.asarray([s], np.float32)).argmax(1)[0])
            s, r, done, _ = env.step(a)
            total += discount * r
            discount *= self.gamma
        env.restore_state(snap)
        return total

    def act(self, state, training=True, env=None):
        if training or env is None:
            return super().act(state, training=training)
        best_a, best_v = 0, -np.inf
        for a in range(self.n_actions):
            v = self._lookahead_value(env, a)
            if v > best_v:
                best_v, best_a = v, a
        return best_a
