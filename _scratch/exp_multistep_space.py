"""多步空间规划实验：rollout 每步都累计 flood fill 空间，对比仅首步空间。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_safe_agent import CNNSafeAgent, flood_fill


class MultiSpaceAgent(CNNSafeAgent):
    def _multi_value(self, env, first_action):
        snap = env.clone_state()
        total, discount = 0.0, 1.0
        s, r, done, _ = env.step(first_action)
        sp = flood_fill(env.body[0], set(env.body[1:]))
        total += discount * (r + self.space_weight * sp / 100)
        discount *= self.gamma
        for _ in range(self.lookahead - 1):
            if done:
                break
            a = int(self.q.forward(np.asarray([s], np.float32)).argmax(1)[0])
            s, r, done, _ = env.step(a)
            sp = flood_fill(env.body[0], set(env.body[1:]))
            total += discount * (r + self.space_weight * sp / 100)
            discount *= self.gamma
        env.restore_state(snap)
        return total

    def act(self, state, training=True, env=None):
        if training or env is None:
            return super().act(state, training=training)
        best_a, best_v = 0, -np.inf
        for a in range(self.n_actions):
            v = self._multi_value(env, a)
            if v > best_v:
                best_v, best_a = v, a
        return best_a


def run(agent):
    scores, death_len = [], []
    for seed in range(5000, 5015):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            a = agent.act(s, training=False, env=env)
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body)); break
    return np.mean(scores), np.max(scores), np.mean(death_len)


for w in (6.0, 12.0):
    base = CNNSafeAgent(space_weight=w); base.load(os.path.join("cnn_ckpt", "best.npz"))
    multi = MultiSpaceAgent(space_weight=w); multi.load(os.path.join("cnn_ckpt", "best.npz"))
    bm = run(base); mm = run(multi)
    print(f"权重{w} 仅首步={bm[0]:.1f}/{bm[1]} 多步空间={mm[0]:.1f}/{mm[1]}")
