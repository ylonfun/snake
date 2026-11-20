"""Dyna 闭环训练器：规划行为 + 真实经验回灌，让 Q 网络自我提升。

从 CNN-DQN best 加载基础 Q，行为策略以小 ε 探索、否则用多步空间规划选动作；
在真实环境执行后按标准 DQN 更新。规划与 Q 共享同一网络，Q 变强则规划变强。
评估用单步贪心（不含规划），专门检验 Q 网络本身的进步。独立 dyna_ckpt 存档。
"""
import os, time
import numpy as np

from rl_env import SnakeEnv
from cnn_safe_agent import CNNMultiSpaceAgent

CKPT = "dyna_ckpt"
INTERVAL = 20000


def evaluate_single(agent, n=30):
    scores = []
    for seed in range(5000, 5000 + n):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            s, _, done, info = env.step(agent.act(s, training=False))
            if done:
                scores.append(info["score"]); break
    return scores


def dyna_act(agent, env, state):
    agent.update_eps()
    if np.random.random() < agent.eps:
        return np.random.randint(4)
    return agent.act(state, training=False, env=env)


def train(total_steps=120000):
    os.makedirs(CKPT, exist_ok=True)
    env = SnakeEnv(state_mode="grid")
    agent = CNNMultiSpaceAgent(lookahead=10, space_weight=8.0,
                               eps_start=0.15, eps_end=0.05, eps_decay_steps=20000)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    agent.step_count = 0
    state = env.reset()
    wall = time.time()

    while agent.step_count < total_steps:
        action = dyna_act(agent, env, state)
        s2, r, done, _ = env.step(action)
        agent.remember(state, action, r, s2, done)
        agent.learn()
        state = env.reset() if done else s2

        if agent.step_count and agent.step_count % INTERVAL == 0:
            scores = evaluate_single(agent)
            agent.save(os.path.join(CKPT, "latest"))
            print(f"步={agent.step_count} 单步30局均分={np.mean(scores):.1f} "
                  f"峰值={max(scores)} eps={agent.eps:.3f} "
                  f"耗时={time.time()-wall:.0f}s")

    agent.save(os.path.join(CKPT, "final"))
    print("Dyna 训练完成")


if __name__ == "__main__":
    train()
