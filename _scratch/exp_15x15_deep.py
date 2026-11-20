"""15×15 深度影响：depth15 vs depth10（beam2，各 2 局）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

for depth in (15, 10):
    agent = CNNBeamAgent(depth=depth, beam=2, space_weight=8.0, tail_pen=5.0)
    for seed in (5000, 5001):
        env = SnakeEnv(rows=15, cols=15, state_mode="grid", deadline_growth=1.0)
        s = env.reset(seed); t0 = time.time(); nsteps = 0
        while True:
            a = agent.act(s, training=False, env=env)
            nsteps += 1
            s, _, done, info = env.step(a)
            if done:
                print(f"depth{depth} seed{seed} 吃{info['score']} 步数{nsteps} "
                      f"每苹果={nsteps/max(1,info['score']):.1f} "
                      f"死因={env.death_cause} 耗时={time.time()-t0:.0f}s")
                break
