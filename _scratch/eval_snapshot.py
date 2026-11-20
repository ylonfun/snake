import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_dqn_agent import CNNDQNAgent

agent = CNNDQNAgent()
agent.load(os.path.join("_scratch", "snapshot_340k.npz"))
scores = []
for seed in range(5000, 5030):
    env = SnakeEnv(state_mode="grid")
    s = env.reset(seed)
    while True:
        s, _, done, info = env.step(agent.act(s, training=False))
        if done:
            scores.append(info["score"]); break
scores = np.array(scores)
print("34万步模型 30 局 均分:", round(scores.mean(), 1),
      "标准差:", round(scores.std(), 1), "峰值:", scores.max())
print("(20万步模型 30 局均分 15.0)")
# 模拟"只抽10局"的均值波动范围：重复抽样看均值散布
means = [np.mean(np.random.choice(scores, 10, replace=False)) for _ in range(200)]
print("同样30局里随机抽10局，均值波动范围:",
      round(min(means), 1), "-", round(max(means), 1))
