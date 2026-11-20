"""冒烟：产品化 CNNMultiSpaceAgent 加载与运行。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from rl_env import SnakeEnv
from cnn_safe_agent import CNNMultiSpaceAgent

agent = CNNMultiSpaceAgent()
agent.load(os.path.join("cnn_ckpt", "best.npz"))
for seed in range(5000, 5003):
    env = SnakeEnv(state_mode="grid")
    s = env.reset(seed)
    while True:
        a = agent.act(s, training=False, env=env)
        s, _, done, info = env.step(a)
        if done:
            print("seed", seed, "分", info["score"], "体长", len(env.body))
            break
