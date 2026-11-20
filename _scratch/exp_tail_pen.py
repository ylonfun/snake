"""tail_pen 强度扫描（15 局），记录分数与死因。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent


def run(tail_pen, n=15):
    agent = CNNBeamAgent(depth=10, beam=3, space_weight=8.0, tail_pen=tail_pen)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, causes = [], {}
    t0 = time.time()
    for seed in range(5000, 5000 + n):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            a = agent.act(s, training=False, env=env)
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"])
                causes[env.death_cause] = causes.get(env.death_cause, 0)+1
                break
    return np.mean(scores), np.max(scores), causes, (time.time()-t0)/n


for tp in (2.0, 3.0, 5.0):
    m, mx, causes, sec = run(tp)
    print(f"tail_pen{tp} 均分={m:.1f} 峰值={mx} 死因={causes} 每局={sec:.1f}s")
