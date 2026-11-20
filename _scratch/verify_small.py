import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent

env = SnakeEnv()
agent = DQNAgent(eps_decay_steps=15000)
state = env.reset(0)
ep_rewards, ep_scores, tds = [], [], []
cur_r = 0.0
next_log = 2000

for t in range(20000):
    action = agent.act(state)
    s2, r, done, info = env.step(action)
    agent.remember(state, action, r, s2, done)
    td = agent.learn()
    if td is not None:
        tds.append(td)
    cur_r += r
    state = s2
    if done:
        ep_rewards.append(cur_r)
        ep_scores.append(info["score"])
        cur_r = 0.0
        state = env.reset()
    if t + 1 >= next_log:
        print(f"步={t+1} eps={agent.eps:.3f} "
              f"近10局奖励={np.mean(ep_rewards[-10:]):.1f} "
              f"近10局分数={np.mean(ep_scores[-10:]):.1f} "
              f"平均TD={np.mean(tds[-200:]):.2f}")
        next_log += 2000
