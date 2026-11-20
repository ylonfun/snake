import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint, CKPT_DIR

agent = DQNAgent(eps_decay_steps=80000)
load_checkpoint(agent, os.path.join(CKPT_DIR, "final.npz"))

# 收集一批真实 next_state（纯贪心 rollout）
next_states = []
env = SnakeEnv()
for seed in range(10):
    s = env.reset(seed)
    for _ in range(120):
        a = agent.act(s, training=False)
        s2, _, done, _ = env.step(a)
        next_states.append(s2)
        s = s2
        if done:
            break
ns = np.asarray(next_states, np.float32)

# 朴素 DQN：目标网络 max
naive = agent.q_target.forward(ns).max(1)
# Double DQN：在线网络选动作，目标网络评估
acts = agent.q.forward(ns).argmax(1)
double = agent.q_target.forward(ns)[np.arange(len(ns)), acts]

print(f"Q值范围: 目标网络 max 均值={naive.mean():.2f}")
print(f"Double 估计均值={double.mean():.2f}")
print(f"max 高估量(朴素-Double)={(naive-double).mean():.2f}")
print(f"当前 Q 值整体分布: min={naive.min():.1f} max={naive.max():.1f}")
