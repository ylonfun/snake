import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_lookahead_agent import CNNLookaheadAgent

agent = CNNLookaheadAgent(lookahead=5)
agent.load(os.path.join("cnn_ckpt", "best.npz"))


def count_safe(env):
    """当前状态下，试每个动作是否不会立即死亡（用克隆，不影响真实环境）。"""
    n_safe = 0
    for a in range(4):
        snap = env.clone_state()
        _, _, done, _ = env.step(a)
        if not done:
            n_safe += 1
        env.restore_state(snap)
    return n_safe


def run(use_lookahead):
    scores, death_len, causes, safe_left = [], [], [], []
    for seed in range(5000, 5030):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            if use_lookahead:
                a = agent.act(s, training=False, env=env)
            else:
                a = agent.act(s, training=False)
            safe = count_safe(env)
            s, r, done, info = env.step(a)
            if done:
                scores.append(info["score"])
                death_len.append(len(env.body))
                causes.append(env.death_cause)
                safe_left.append(safe)
                break
    return np.array(scores), np.array(death_len), causes, np.array(safe_left)


for name, use in [("单步贪心(基线)", False), ("5步前瞻(W4)", True)]:
    sc, dl, causes, safe = run(use)
    print("=" * 50)
    print(name)
    print("  均分:", round(sc.mean(), 1), "峰值:", sc.max(),
          "死亡体长均值:", round(dl.mean(), 1))
    print("  死因:", {c: causes.count(c) for c in set(causes)})
    print("  真困死(0安全)占比:", round((safe == 0).mean() * 100), "%")
