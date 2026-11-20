"""15×15 目标导向：food_weight 对比（depth10 beam2，各 3 局）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

for fw in (0.0, 3.0):
    agent = CNNBeamAgent(depth=10, beam=2, space_weight=8.0,
                         tail_pen=5.0, food_weight=fw)
    scores, per_apple = [], []
    for seed in range(5000, 5003):
        env = SnakeEnv(rows=15, cols=15, state_mode="grid", deadline_growth=1.0)
        s = env.reset(seed); t0 = time.time(); nsteps = 0
        while True:
            a = agent.act(s, training=False, env=env)
            nsteps += 1
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"])
                per_apple.append(nsteps/max(1, info["score"]))
                print(f"fw{fw} seed{seed} 吃{info['score']} 步数{nsteps} "
                      f"每苹果={nsteps/max(1,info['score']):.1f} "
                      f"死因={env.death_cause} 耗时={time.time()-t0:.0f}s")
                break
    print(f"== fw{fw} 均分={np.mean(scores):.1f} 每苹果={np.mean(per_apple):.1f}\n")
