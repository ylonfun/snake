"""多步空间 × lookahead 深度扫描（15 局）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_safe_agent import CNNMultiSpaceAgent


def run(lookahead):
    agent = CNNMultiSpaceAgent(lookahead=lookahead, space_weight=8.0)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, death_len, ms = [], [], []
    for seed in range(5000, 5015):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        t0, n = time.time(), 0
        while True:
            a = agent.act(s, training=False, env=env)
            n += 1
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body))
                ms.append((time.time()-t0)/n*1000); break
    return np.mean(scores), np.max(scores), np.mean(death_len), np.mean(ms)


for la in (10, 15, 20):
    m, mx, dl, ms = run(la)
    print(f"lookahead{la:2d} 均分={m:.1f} 峰值={mx} 死亡体长={dl:.1f} 每决策={ms:.1f}ms")
