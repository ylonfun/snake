"""rand 修复后 10×10 10 局评估（depth8 beam2 动态期限）。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

agent = CNNBeamAgent(depth=8, beam=2, space_weight=8.0, tail_pen=5.0)
scores = []
for seed in range(5000, 5010):
    env = SnakeEnv(state_mode="grid", deadline_growth=1.0)
    s = env.reset(seed); nsteps = 0
    while True:
        a = agent.act(s, training=False, env=env)
        nsteps += 1
        s, _, done, info = env.step(a)
        if done:
            scores.append(info["score"])
            print(f"seed{seed} 吃{info['score']} 死因={env.death_cause}")
            break
print("10局均分=", np.mean(scores), "峰值=", max(scores))
