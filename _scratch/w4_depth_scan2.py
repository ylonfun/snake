"""W4 深度扫描扩展：K=10/15/20/25，看趋势与上限。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_lookahead_agent import CNNLookaheadAgent

N_GAMES = 12


def run(depth):
    agent = CNNLookaheadAgent(lookahead=depth)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, death_len, ms = [], [], []
    for seed in range(5000, 5000 + N_GAMES):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        t0 = time.time()
        n_act = 0
        while True:
            a = agent.act(s, training=False, env=env)
            n_act += 1
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body))
                ms.append((time.time() - t0) / n_act * 1000)
                break
    return np.mean(scores), np.max(scores), np.mean(death_len), np.mean(ms)


for d in (10, 15, 20, 25):
    m, mx, dl, ms = run(d)
    print(f"深度{d:2d} 均分={m:.1f} 峰值={mx} 死亡体长={dl:.1f} 每决策={ms:.1f}ms")
