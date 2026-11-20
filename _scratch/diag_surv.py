import os, time

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent

TRAIN_STEPS = 100000


def train_group(min_len):
    env = SnakeEnv(survival_mode=True, init_len=25, min_len=min_len)
    agent = DQNAgent(state_mode="survival", eps_decay_steps=60000)
    state = env.reset()
    while agent.step_count < TRAIN_STEPS:
        a = agent.act(state)
        s2, r, done, info = env.step(a)
        agent.remember(state, a, r, s2, done)
        agent.learn()
        state = s2
        if done:
            state = env.reset()
    return agent, min_len


def eval_group(agent, min_len):
    lifes, lens, sns = [], [], []
    for seed in range(5000, 5015):
        e = SnakeEnv(survival_mode=True, init_len=25, min_len=min_len)
        st = e.reset(seed)
        moves, sn = 0, 0
        while True:
            a = agent.act(st, training=False)
            if a == 4:
                sn += 1
            st, r, done, info = e.step(a)
            moves += 1
            if done:
                lifes.append(e.steps); lens.append(len(e.body))
                sns.append(100 * sn / moves)
                break
    return np.mean(lifes), np.mean(lens), np.mean(sns)


for label, ml in [("允许缩身(下限10)", 10), ("不能缩身(下限25)", 25)]:
    t0 = time.time()
    agent, min_len = train_group(ml)
    life, length, sn = eval_group(agent, min_len)
    print(f"{label}: 平均寿命={life:.0f}步 平均长度={length:.1f} "
          f"shrink占比={sn:.1f}% 耗时={time.time()-t0:.0f}s", flush=True)
