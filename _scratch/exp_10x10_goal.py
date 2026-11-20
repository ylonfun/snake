"""10×10 目标导向短路径：food_weight 扫描（depth10 beam2，各 5 局）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent


def run(fw, n=5):
    agent = CNNBeamAgent(depth=10, beam=2, space_weight=8.0,
                         tail_pen=5.0, food_weight=fw)
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
    return (np.mean(scores), np.max(scores), np.mean(per_apple),
            (time.time()-t0)/n)


for fw in (0.0, 2.0, 3.0):
    m, mx, pa, sec = run(fw)
    print(f"fw{fw} 均分={m:.1f} 峰值={mx} 每苹果={pa:.1f}步 每局={sec:.1f}s")
