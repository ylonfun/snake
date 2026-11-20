"""产品化 beam search 30 局确认（depth10 beam3）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

agent = CNNBeamAgent(depth=10, beam=3, space_weight=8.0)
agent.load(os.path.join("cnn_ckpt", "best.npz"))
scores, death_len, steps = [], [], []
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
            steps.append(n); break
print(f"30局 均分={np.mean(scores):.1f} 中位数={np.median(scores):.1f} "
      f"峰值={max(scores)} 死亡体长={np.mean(death_len):.1f} "
      f"步数={np.mean(steps):.0f} 耗时={time.time()-wall:.0f}s")
