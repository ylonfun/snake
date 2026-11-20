"""隐空间世界模型：编码器 E + 动力学 D + 奖励头 R + 解码器 Dec + 终止头 C。

观测 o 经 E 压缩为有界隐状态 z（tanh）；D 由 (z, 动作) 预测下一隐状态；
R 预测奖励；Dec 重构观测；C（sigmoid）预测 episode 是否继续（1=继续,0=终止）。
稳定性：目标编码器 E_t（EMA）、隐状态 tanh 限幅、梯度裁剪、较低学习率。
"""
import numpy as np

from mlp import MLP

DTYPE = np.float32


class LatentWorldModel:
    def __init__(self, state_dim=42, n_actions=4, z_dim=32, lr=3e-4,
                 ema_tau=0.02, done_weight=30.0, r_lo=-10.0, r_hi=10.0,
                 r_food=10.0, food_r_weight=1.0):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.z_dim = z_dim
        self.ema_tau = ema_tau
        self.done_weight = done_weight
        self.r_lo, self.r_hi = r_lo, r_hi   # 真实奖励的已知边界（死亡-10、吃果+10）
        self.r_food = r_food
        self.food_r_weight = food_r_weight   # 稀疏吃苹果样本的损失加权
        self.E = MLP([state_dim, 64, z_dim], lr, out_act="tanh")
        self.D = MLP([z_dim + n_actions, 64, z_dim], lr, out_act="tanh")
        self.R = MLP([z_dim, 64, 1], lr)
        self.C = MLP([z_dim, 16, 1], lr, out_act="sigmoid")
        self.Dec = MLP([z_dim, 64, state_dim], lr)
        self.E_t = MLP([state_dim, 64, z_dim], lr, out_act="tanh")
        self.E_t.W = [w.copy() for w in self.E.W]
        self.E_t.b = [x.copy() for x in self.E.b]

    def decode(self, z):
        return self.Dec.forward(z)

    def cont_prob(self, z):
        return self.C.forward(z)[:, 0]

    def _onehot(self, actions):
        oh = np.zeros((len(actions), self.n_actions), DTYPE)
        oh[np.arange(len(actions)), actions] = 1.0
        return oh

    def encode(self, obs):
        return self.E.forward(np.asarray(obs, DTYPE))

    def step(self, z, actions):
        x = np.concatenate([z, self._onehot(actions)], axis=1)
        return self.D.forward(x)

    def reward(self, z):
        # 规划/想象用：裁剪到真实奖励边界，分布外隐状态也不会产生虚高奖励
        return np.clip(self.R.forward(z), self.r_lo, self.r_hi)[:, 0]

    def update_world(self, o, a, r, o2, done):
        n = len(o)
        z = self.E.forward(o)
        z_target = self.E_t.forward(o2)
        x = np.concatenate([z, self._onehot(a)], axis=1)
        z_pred = self.D.forward(x)
        r_pred = self.R.forward(z_pred)
        c_pred = self.C.forward(z_pred)

        # 奖励头：吃苹果样本稀疏，加权防止 R 向整体均值收缩、低估 +10
        r_col = r.reshape(-1, 1)
        r_w = np.where(r_col >= self.r_food * 0.5, self.food_r_weight, 1.0)
        g_from_R = self.R.backward(2 * (r_pred - r_col) * r_w / n)
        # 终止头：sigmoid + 二元交叉熵，目标 = 1 - done
        c_target = (1.0 - done).reshape(-1, 1)
        w = np.where(done.reshape(-1, 1) == 1, self.done_weight, 1.0)
        g_from_C = self.C.backward((c_pred - c_target) * w / n)
        # 动力学：模型损失 + 奖励/终止回传
        g_z = 2 * (z_pred - z_target) / n + g_from_R + g_from_C
        g_x = self.D.backward(g_z)[:, :self.z_dim]
        self.E.backward(g_x)

        # 解码器独立训练（z 固定，不回传 E）
        dec_pred = self.Dec.forward(z)
        self.Dec.backward(2 * (dec_pred - o) / n)

        for net in (self.E, self.D, self.R, self.C, self.Dec):
            net.clip_grads(5.0)
            net.adam_step()

        # 目标编码器 EMA
        for i in range(len(self.E_t.W)):
            self.E_t.W[i] = (1 - self.ema_tau) * self.E_t.W[i] + self.ema_tau * self.E.W[i]
            self.E_t.b[i] = (1 - self.ema_tau) * self.E_t.b[i] + self.ema_tau * self.E.b[i]

    def rollout(self, init_z, actions):
        z = np.asarray(init_z, DTYPE)
        traj = [z]
        for a in actions:
            z = self.step(np.asarray([z]), np.asarray([a]))[0]
            traj.append(z)
        return np.array(traj)
