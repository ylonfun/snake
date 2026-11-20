"""阶段二：加载冻结世界模型，在固定想象环境中专训 actor-critic。

分阶段存档：每 interval 步保存一个独立 checkpoint（stage_0 ... stage_9），
配合菜单里的"阶段浏览器"回看不同训练阶段；评估使用随机局面（随机种子），
检验在未见过的苹果位置上的临场决策与泛化能力，而非固定重播。
"""
import os
import time
import random as pyrandom

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

CKPT_DIR = "dreamer_ckpt"


def evaluate(agent, n_games=10, seeds=None):
    # 默认随机局面：每次评估随机抽种子，苹果位置随机，检验随机应变
    if seeds is None:
        seeds = [pyrandom.randint(10000, 99999) for _ in range(n_games)]
    scores, steps_list = [], []
    for seed in seeds:
        env = SnakeEnv(); st = env.reset(seed)
        while True:
            st, _, done, info = env.step(agent.act(st, training=False))
            if done:
                scores.append(info["score"]); steps_list.append(info["steps"]); break
    return scores, steps_list


def train(total_steps=200000, interval=20000):
    os.makedirs(CKPT_DIR, exist_ok=True)
    agent = DreamerAgent(freeze_world=True)
    agent.load_world("world_frozen.npz")
    env = SnakeEnv(); st = env.reset()
    next_mark = interval
    stage = 0
    wall = time.time()
    while agent.step_count < total_steps:
        a = agent.act(st, training=True)
        s2, r, done, _ = env.step(a)
        agent.remember(st, a, r, s2, done)
        agent.learn()
        st = s2
        if done:
            st = env.reset()
        if agent.step_count >= next_mark:
            sc, sp = evaluate(agent)
            agent.save(os.path.join(CKPT_DIR, f"stage_{stage}"))   # 阶段独立存档
            agent.save(os.path.join(CKPT_DIR, "latest"))
            print(f"阶段{stage} 步={agent.step_count} 随机局均分={np.mean(sc):.1f} "
                  f"峰值={max(sc)} 步数={np.mean(sp):.0f} 耗时={time.time()-wall:.0f}s")
            stage += 1
            next_mark += interval
    agent.save(os.path.join(CKPT_DIR, "final"))
    print("完成")


if __name__ == "__main__":
    train()
