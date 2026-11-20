import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_dqn_agent import CNNDQNAgent

env = SnakeEnv(state_mode="grid")
agent = CNNDQNAgent(learn_start=50)
s = env.reset(0)
out = agent.q.forward(np.asarray([s]))
print("CNN 输出形状:", out.shape)
for i in range(200):
    a = agent.act(s, training=True)
    s2, r, done, info = env.step(a)
    agent.remember(s, a, r, s2, done)
    agent.learn()
    s = s2
    if done:
        s = env.reset()
print("200 步训练无报错，buffer 大小", agent.buffer.size)
print("贪心动作:", agent.act(s, training=False))
