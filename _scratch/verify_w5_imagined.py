"""30 局对比：单步贪心 / W4 环境克隆前瞻 / W5 内部模型想象前瞻。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_lookahead_agent import CNNLookaheadAgent
from cnn_imagined_agent import CNNImaginedAgent

w5 = CNNImaginedAgent(lookahead=5)
w5.load(os.path.join("cnn_ckpt", "best.npz"))
w4 = CNNLookaheadAgent(lookahead=5)
w4.load(os.path.join("cnn_ckpt", "best.npz"))


def count_safe(env):
    n_safe = 0
    for a in range(4):
        snap = env.clone_state()
        _, _, done, _ = env.step(a)
        if not done:
            n_safe += 1
        env.restore_state(snap)
    return n_safe


def run(mode):
    scores, death_len, causes, safe_left = [], [], [], []
    for seed in range(5000, 5030):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            if mode == "single":
                a = w5.act(s, training=False)
            elif mode == "w4":
                a = w4.act(s, training=False, env=env)
            else:
                a = w5.act(s, training=False, env=env)
            safe = count_safe(env)
            s, r, done, info = env.step(a)
            if done:
                scores.append(info["score"])
                death_len.append(len(env.body))
                causes.append(env.death_cause)
                safe_left.append(safe)
                break
    return np.array(scores), np.array(death_len), causes, np.array(safe_left)


for name, mode in [("单步贪心", "single"), ("W4 环境克隆", "w4"),
                   ("W5 内部模型", "w5")]:
    sc, dl, causes, safe = run(mode)
    print("=" * 50)
    print(name)
    print("  均分:", round(sc.mean(), 1), "峰值:", sc.max(),
          "死亡体长均值:", round(dl.mean(), 1))
    print("  死因:", {c: causes.count(c) for c in set(causes)})
    print("  真困死(0安全)占比:", round((safe == 0).mean() * 100), "%")
