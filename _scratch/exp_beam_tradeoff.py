"""动态期限下 beam 配置权衡（每配置 10 局）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent


def run(depth, beam, n=10):
    agent = CNNBeamAgent(depth=depth, beam=beam, space_weight=8.0, tail_pen=5.0)
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
    return (np.mean(scores), np.max(scores), (time.time()-t0)/n,
            np.mean(per_apple))


for depth, beam in ((8, 2), (10, 2), (10, 3)):
    m, mx, sec, pa = run(depth, beam)
    print(f"depth{depth} beam{beam} 均分={m:.1f} 峰值={mx} "
          f"每局={sec:.1f}s 每苹果={pa:.1f}步")
