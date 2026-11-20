import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_dqn_agent import CNNDQNAgent

agent = CNNDQNAgent()
agent.load(os.path.join("cnn_ckpt", "final.npz"))
scores, steps, dlen, causes = [], [], [], []
for seed in range(5000, 5030):
    env = SnakeEnv(state_mode="grid")
    s = env.reset(seed)
    while True:
        s, r, done, info = env.step(agent.act(s, training=False))
        if done:
            scores.append(info["score"]); steps.append(info["steps"])
            dlen.append(len(env.body)); causes.append(env.death_cause)
            break
scores, steps, dlen = np.array(scores), np.array(steps), np.array(dlen)
print("局数:", len(scores))
print("均分:", round(scores.mean(), 1), "峰值:", scores.max())
print("平均步数:", round(steps.mean()), "最大步数:", steps.max())
print("死亡体长 平均:", round(dlen.mean(), 1), "中位:", np.median(dlen),
      "最大:", dlen.max())
from collections import Counter
print("死因分布:", dict(Counter(causes)))
