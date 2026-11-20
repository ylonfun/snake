import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint, CKPT_DIR

env = SnakeEnv()
state = env.reset(0)
print("当前 local 状态维度:", state.shape)
agent = DQNAgent()
print("旧 DQN 网络输入维度:", agent.q.sizes[0])
path = os.path.join(CKPT_DIR, "final.npz")
print("旧 DQN 模型存在:", os.path.exists(path))
load_checkpoint(agent, path)
try:
    agent.act(state, training=False)
    print("直接喂完整状态：正常")
except Exception as e:
    print("直接喂完整状态报错:", type(e).__name__, str(e)[:80])
# 对比：切片到 40 维
try:
    agent.act(state[:40], training=False)
    print("切片 state[:40]：正常")
except Exception as e:
    print("切片仍报错:", e)
