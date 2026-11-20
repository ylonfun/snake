"""Dreamer 式智能体：在隐空间世界模型中想象轨迹，用 actor-critic 学习策略。

- 世界模型（LatentWorldModel）由真实 transition 训练，含终止头 C；
- actor π：隐状态 -> 动作 logits（softmax 采样/贪心）；
- critic V：隐状态 -> 价值；
- 想象时由 C 判断是否终止（终止后未来回报归零），再用 λ-return 解析计算目标
  （stop-gradient，不做时序 BPTT）：critic 回归 λ-return，actor 用 REINFORCE。
实现 BaseAgent 接口，可接入统一训练器。
"""
import numpy as np

from base_agent import BaseAgent
from replay_buffer import ReplayBuffer
from latent_world_model import LatentWorldModel
from mlp import MLP

DTYPE = np.float32


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


class DreamerAgent(BaseAgent):
    def __init__(self, state_dim=42, n_actions=4, z_dim=32, gamma=0.95,
                 lam=0.9, horizon=5, imag_batch=16, batch_size=64,
                 lr=3e-4, buffer_capacity=100000, learn_start=1000,
                 ent_coef=0.02, freeze_world=False):
        self.state_dim, self.n_actions = state_dim, n_actions
        self.z_dim, self.gamma, self.lam = z_dim, gamma, lam
        self.horizon, self.imag_batch, self.batch_size = horizon, imag_batch, batch_size
        self.learn_start = learn_start
        self.ent_coef = ent_coef
        self.freeze_world = freeze_world
        self.world = LatentWorldModel(state_dim, n_actions, z_dim, lr)
        self.actor = MLP([z_dim, 64, n_actions], lr)
        self.critic = MLP([z_dim, 64, 1], lr)
        self.buffer = ReplayBuffer(buffer_capacity, state_dim)
        self.step_count = 0

    # 动作 -> 该正方向在 _VISION_DIRS 中的射线索引；动作的反向配对
    _ACTION_RAY = (0, 4, 6, 2)
    _OPPOSITE = (1, 0, 3, 2)

    def _legal_mask(self, state):
        """根据射线特征判断每个动作执行后是否会立即撞墙/撞硬身体（True=安全）。

        利用状态里已有的两个信号：正方向 1/距离=1 表示紧邻墙，adj_self=1 表示
        紧邻不会移走的硬身体。并模拟环境对 180 度反向的钳制。
        """
        cur_a = int(np.asarray(state[:4]).argmax())   # 当前头方向动作
        safe = np.ones(self.n_actions, dtype=bool)
        for a in range(self.n_actions):
            eff_a = cur_a if a == self._OPPOSITE[cur_a] else a  # 反向被钳成直行
            base = 8 + self._ACTION_RAY[eff_a] * 4
            if state[base] >= 1.0 or state[base + 3] >= 1.0:
                safe[a] = False
        return safe

    def act(self, state, training=True):
        z = self.world.encode(np.asarray([state], DTYPE))
        logits = self.actor.forward(z)[0]
        legal = self._legal_mask(np.asarray(state, DTYPE))
        if not legal.any():                          # 兜底：物理困死时不改变原选择
            legal = np.ones(self.n_actions, dtype=bool)
        if training:
            p = np.where(legal, softmax(logits[None])[0], 0.0)   # 非法动作概率归零
            return int(np.random.choice(self.n_actions, p=p / p.sum()))
        return int(np.where(legal, logits, -1e9).argmax())       # 安全动作里选最优

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        self.step_count += 1
        if self.buffer.size < self.learn_start:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        if not self.freeze_world:
            self.world.update_world(s, a, r, s2, d)
        self._imagine_and_update(s[:self.imag_batch])
        return None

    def load_world(self, path):
        data = np.load(path)
        for tag, net in [("E", self.world.E), ("D", self.world.D), ("R", self.world.R),
                         ("C", self.world.C), ("Dec", self.world.Dec), ("Et", self.world.E_t)]:
            for i in range(len(net.W)):
                net.W[i] = data[f"{tag}_W{i}"].copy()
                net.b[i] = data[f"{tag}_b{i}"].copy()

    def _action_values(self, z, depth=None):
        """对每个动作在世界模型中多步想象，返回该动作的折扣回报（行向量）。"""
        depth = depth or self.horizon
        q = np.zeros((1, self.n_actions))
        for a in range(self.n_actions):
            cur = self.world.step(z, np.asarray([a]))
            r1 = self.world.reward(cur)[0]
            c1 = self.world.C.forward(cur)[0, 0]
            if c1 < 0.5:
                q[0, a] = r1
                continue
            G, disc, cur_z = r1, self.gamma, cur
            for t in range(depth - 1):
                a2 = int(self.actor.forward(cur_z).argmax(1)[0])
                cur_z = self.world.step(cur_z, np.asarray([a2]))
                rr = self.world.reward(cur_z)[0]
                cc = self.world.C.forward(cur_z)[0, 0]
                G += disc * rr
                disc *= self.gamma
                if cc < 0.5:
                    break
            else:
                G += disc * self.critic.forward(cur_z)[0, 0]
            q[0, a] = G
        return q

    def _imagine_and_update(self, init_states):
        H = self.horizon
        K = len(init_states)
        crit_z, crit_t = [], []

        # 1) 用当前 actor 采样想象，得到 λ-return 目标（训练 critic）
        for k in range(K):
            cur = self.world.encode(np.asarray([init_states[k]]))
            zs, aa, rr, cc = [cur[0]], [], [], []
            for t in range(H):
                a = int(np.random.choice(self.n_actions,
                                         p=softmax(self.actor.forward(cur))[0]))
                z_next = self.world.step(cur, np.asarray([a]))
                rr.append(self.world.reward(z_next)[0])
                cp = self.world.C.forward(z_next)[0, 0]
                aa.append(a); cc.append(cp); zs.append(z_next[0]); cur = z_next
                if cp < 0.5:
                    break
            T = len(rr)
            terminated = cc[-1] < 0.5
            vals = [self.critic.forward(np.asarray([zz]))[0, 0] for zz in zs]
            G = 0.0 if terminated else vals[T]
            for t in range(T - 1, -1, -1):
                nxt = 0.0 if (t == T - 1 and terminated) else vals[t + 1]
                G = rr[t] + self.gamma * ((1 - self.lam) * nxt + self.lam * G)
                crit_z.append(zs[t]); crit_t.append(G)

        # 2) critic 回归 λ-return
        self.critic.zero_grad()
        for zz, tt in zip(crit_z, crit_t):
            pred = self.critic.forward(np.asarray([zz]))
            self.critic.backward(2 * (pred - np.asarray([[tt]])))
        sc = 1.0 / len(crit_z)
        for i in range(len(self.critic.W)):
            self.critic.gW[i] *= sc
            self.critic.gb[i] *= sc
        self.critic.clip_grads(5.0); self.critic.adam_step()

        # 3) actor：软 MPC 梯度，最大化各动作想象 Q 的期望（不依赖采样，符号可靠）
        self.actor.zero_grad()
        for k in range(K):
            z = self.world.encode(np.asarray([init_states[k]]))
            Q = self._action_values(z)
            p = softmax(self.actor.forward(z))
            meanQ = (p * Q).sum(1, keepdims=True)
            d = p * (meanQ - Q)
            if self.ent_coef > 0:
                logp = np.log(p + 1e-8)
                Hent = -(p * logp).sum(1, keepdims=True)
                d += self.ent_coef * p * (logp + Hent)
            self.actor.backward(d / K)
        self.actor.clip_grads(5.0); self.actor.adam_step()

    def _nets(self):
        w = self.world
        return [("E", w.E), ("D", w.D), ("R", w.R), ("C", w.C),
                ("Dec", w.Dec), ("Et", w.E_t),
                ("actor", self.actor), ("critic", self.critic)]

    def save(self, path):
        data = {"step": self.step_count}
        for tag, net in self._nets():
            for i in range(len(net.W)):
                data[f"{tag}_W{i}"] = net.W[i]
                data[f"{tag}_b{i}"] = net.b[i]
        np.savez(path, **data)

    def load(self, path):
        data = np.load(path)
        for tag, net in self._nets():
            for i in range(len(net.W)):
                net.W[i] = data[f"{tag}_W{i}"].copy()
                net.b[i] = data[f"{tag}_b{i}"].copy()
        self.step_count = int(data["step"])
