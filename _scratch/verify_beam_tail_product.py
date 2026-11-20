"""产品版 beam+tail 30 局确认。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

agent = CNNBeamAgent(depth=10, beam=3, space_weight=8.0, tail_pen=5.0)
agent.load(os.path.join("cnn_ckpt", "best.npz"))
scores, death_len, steps, causes = [], [], [], {}
wall = time.time()
for seed in range(5000, 5030):
    env = SnakeEnv(state_mode="grid")
    s = env.reset(seed)
    n = 0
    while True:
        a = agent.act(s, training=False, env=env)
        n += 1
        s, _, done, info = env.step(a)
        if done:
            scores.append(info["score"]); death_len.append(len(env.body))
            steps.append(n); causes[env.death_cause] = causes.get(env.death_cause, 0)+1
            break
print(f"30局 均分={np.mean(scores):.1f} 中位数={np.median(scores):.1f} 峰值={max(scores)} "
      f"死亡体长={np.mean(death_len):.1f} 步数={np.mean(steps):.0f} "
      f"死因={causes} 耗时={time.time()-wall:.0f}s")
