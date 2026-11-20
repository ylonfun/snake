"""W6 评估：扫描空间权重，30 局对比 W4 基线。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_safe_agent import CNNSafeAgent


def run(weight):
    agent = CNNSafeAgent(lookahead=10, space_weight=weight)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, death_len = [], []
    for seed in range(5000, 5030):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            a = agent.act(s, training=False, env=env)
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body)); break
    return np.mean(scores), np.max(scores), np.mean(death_len)


for w in (0.0, 1.0, 2.0, 4.0):
    m, mx, dl = run(w)
    tag = "W4基线" if w == 0 else f"W6 权重{w}"
    print(f"{tag:10s} 均分={m:.1f} 峰值={mx} 死亡体长={dl:.1f}")
