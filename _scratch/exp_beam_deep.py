"""更强 beam 配置扫描（每配置 10 局）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent


def run(depth, beam, n=10):
    agent = CNNBeamAgent(depth=depth, beam=beam, space_weight=8.0)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, death_len, t0 = [], [], time.time()
    for seed in range(5000, 5000 + n):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            a = agent.act(s, training=False, env=env)
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body))
                break
    return np.mean(scores), np.max(scores), np.mean(death_len), (time.time()-t0)/n


for depth, beam in ((12, 3), (15, 4)):
    m, mx, dl, sec = run(depth, beam)
    print(f"depth{depth:2d} beam{beam} 均分={m:.1f} 峰值={mx} "
          f"死亡体长={dl:.1f} 每局={sec:.1f}s")
