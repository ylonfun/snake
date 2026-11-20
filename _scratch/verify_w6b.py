"""W6 高权重扫描（15 局快速）：找空间权重最佳点。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_safe_agent import CNNSafeAgent


def run(weight):
    agent = CNNSafeAgent(lookahead=10, space_weight=weight)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, death_len, steps = [], [], []
    for seed in range(5000, 5015):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        n = 0
        while True:
            a = agent.act(s, training=False, env=env)
            n += 1
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body))
                steps.append(n); break
    return np.mean(scores), np.max(scores), np.mean(death_len), np.mean(steps)


for w in (4.0, 6.0, 8.0, 12.0):
    m, mx, dl, st = run(w)
    print(f"权重{w:4.1f} 均分={m:.1f} 峰值={mx} 死亡体长={dl:.1f} 平均步数={st:.0f}")
