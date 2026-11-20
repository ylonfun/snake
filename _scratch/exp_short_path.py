"""短路径：动态期限 depth8 beam2 下 space_weight 扫描（10 局）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent


def run(sw, n=10):
    agent = CNNBeamAgent(depth=8, beam=2, space_weight=sw, tail_pen=5.0)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, per_apple, t0 = [], [], time.time()
    for seed in range(5000, 5000 + n):
        env = SnakeEnv(state_mode="grid", deadline_growth=1.0)
        s = env.reset(seed); nsteps = 0
        while True:
            a = agent.act(s, training=False, env=env)
            nsteps += 1
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"])
                per_apple.append(nsteps/max(1, info["score"]))
                break
    return np.mean(scores), np.max(scores), np.mean(per_apple), (time.time()-t0)/n


for sw in (2.0, 4.0, 8.0):
    m, mx, pa, sec = run(sw)
    print(f"space_weight{sw} 均分={m:.1f} 峰值={mx} 每苹果={pa:.1f}步 每局={sec:.1f}s")
