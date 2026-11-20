"""测量 W4 环境克隆前瞻的实际决策耗时（纯逻辑、无渲染）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_lookahead_agent import CNNLookaheadAgent

for K in [5, 10]:
    agent = CNNLookaheadAgent(lookahead=K)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    env = SnakeEnv(state_mode="grid")
    s = env.reset(6000)
    t0 = time.time()
    n_act = 0
    while True:
        a = agent.act(s, training=False, env=env)
        n_act += 1
        s, r, done, info = env.step(a)
        if done:
            break
    dt = time.time() - t0
    print(f"K={K} 局分={info['score']} 步数={n_act} 总耗时={dt:.2f}s "
          f"每决策={dt/n_act*1000:.1f}ms")
