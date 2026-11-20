import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from cnn_dqn_agent import CNNDQNAgent

agent = CNNDQNAgent(eps_decay_steps=80000)
agent.load(os.path.join("cnn_ckpt", "latest.npz"))
h = np.load(os.path.join("cnn_ckpt", "history.npz"))
if agent.step_count == 0:
    agent.step_count = int(h["step"][-1])
agent.update_eps()
interval = 20000
next_mark = ((agent.step_count // interval) + 1) * interval
print("恢复步数:", agent.step_count, "eps:", round(agent.eps, 3),
      "下一个评估点:", next_mark)
# 确认贪心动作仍正常（模型参数没坏）
env_path = os.path.join(root)
from rl_env import SnakeEnv
env = SnakeEnv(state_mode="grid")
s = env.reset(0)
print("贪心动作:", agent.act(s, training=False))
