import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint, CKPT_DIR

agent = DQNAgent(eps_decay_steps=80000)
load_checkpoint(agent, os.path.join(CKPT_DIR, "final.npz"))

scores, steps_list = [], []
for seed in range(30):
    env = SnakeEnv()
    state = env.reset(seed)
    while True:
        action = agent.act(state, training=False)
        state, _, done, info = env.step(action)
        if done:
            scores.append(info["score"])
            steps_list.append(info["steps"])
            break
scores = np.asarray(scores)
steps_list = np.asarray(steps_list)
print(f"30局 平均={scores.mean():.1f} 最高={scores.max()} 最低={scores.min()}")
print(f"分数分布: {np.sort(scores)}")
print(f"平均步数={steps_list.mean():.0f} 到99局数={(scores==99).sum()}")
