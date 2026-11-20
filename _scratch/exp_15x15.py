"""15×15 地图 beam 规划泛化测试。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

N = 15
agent = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
scores, per_apple, t0 = [], [], time.time()
for seed in range(5000, 5005):
    env = SnakeEnv(rows=N, cols=N, state_mode="grid", deadline_growth=1.0)
    s = env.reset(seed); nsteps = 0
    while True:
        a = agent.act(s, training=False, env=env)
        nsteps += 1
        s, _, done, info = env.step(a)
        if done:
            scores.append(info["score"])
            per_apple.append(nsteps/max(1, info["score"]))
            print(f"seed{seed} 吃{info['score']} 步数{nsteps} 每苹果="
                  f"{nsteps/max(1,info['score']):.1f} 死因={env.death_cause}")
            break
print(f"15x15 均分={np.mean(scores):.1f} 峰值={max(scores)} "
      f"每苹果={np.mean(per_apple):.1f} 每局={(time.time()-t0)/5:.1f}s")
