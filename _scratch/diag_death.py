import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint, CKPT_DIR
from settings import DIRECTIONS

agent = DQNAgent(eps_decay_steps=80000)
load_checkpoint(agent, os.path.join(CKPT_DIR, "final.npz"))

death_wall = death_self = full = 0
death_lengths = []
for seed in range(40):
    env = SnakeEnv()
    state = env.reset(seed)
    while True:
        action = agent.act(state, training=False)
        d = DIRECTIONS[action]
        h = env.body[0]
        nh = (h[0] + d[0], h[1] + d[1])
        state, _, done, info = env.step(action)
        if done:
            if info.get("full"):
                full += 1
            else:
                death_lengths.append(len(env.body))
                if (nh[0] < 0 or nh[0] >= env.cols
                        or nh[1] < 0 or nh[1] >= env.rows):
                    death_wall += 1
                else:
                    death_self += 1
            break

death_lengths = np.asarray(death_lengths)
print(f"撞墙={death_wall} 撞自身={death_self} 吃满={full}")
print(f"死亡时长度 平均={death_lengths.mean():.1f} 最小={death_lengths.min()} "
      f"最大={death_lengths.max()}")
print(f"死亡长度分布: {np.sort(death_lengths)}")
