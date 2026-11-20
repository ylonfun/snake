"""CNN-DQN 训练器：完整 grid 表征 + 卷积网络，记录学习曲线并周期贪心评估。

支持续训：若 cnn_ckpt/latest.npz 存在则加载参数、步数与历史曲线，继续训练到
total_steps。评估默认 30 局（降低小样本噪声），并额外保存 best  checkpoint
（本次运行中评估均分最高的模型），避免峰值被后续震荡覆盖。
"""
import os
import time

import numpy as np

from rl_env import SnakeEnv
from cnn_dqn_agent import CNNDQNAgent

CKPT_DIR = "cnn_ckpt"
EVAL_GAMES = 30


def evaluate(agent, n_games=EVAL_GAMES, base_seed=5000):
    scores, steps_list = [], []
    for seed in range(base_seed, base_seed + n_games):
        env = SnakeEnv(state_mode="grid")
        state = env.reset(seed)
        while True:
            state, _, done, info = env.step(agent.act(state, training=False))
            if done:
                scores.append(info["score"])
                steps_list.append(info["steps"])
                break
    return scores, steps_list


def train(total_steps=400000, interval=20000):
    os.makedirs(CKPT_DIR, exist_ok=True)
    env = SnakeEnv(state_mode="grid")
    agent = CNNDQNAgent(eps_decay_steps=80000)
    history = {"step": [], "eval_score": [], "eval_steps": []}
    latest = os.path.join(CKPT_DIR, "latest.npz")
    if os.path.exists(latest):
        agent.load(latest)
        old = np.load(os.path.join(CKPT_DIR, "history.npz"))
        for k in history:
            history[k] = list(old[k])
        if agent.step_count == 0 and len(history["step"]):
            agent.step_count = int(history["step"][-1])  # 旧存档无 step，用曲线进度
        print(f"检测到存档，从 {agent.step_count} 步续训")
    state = env.reset()
    next_mark = ((agent.step_count // interval) + 1) * interval
    best_score = -np.inf
    wall = time.time()

    while agent.step_count < total_steps:
        action = agent.act(state)
        s2, r, done, _ = env.step(action)
        agent.remember(state, action, r, s2, done)
        agent.learn()
        state = s2
        if done:
            state = env.reset()

        if agent.step_count >= next_mark:
            scores, steps = evaluate(agent)
            mean_score = float(np.mean(scores))
            history["step"].append(agent.step_count)
            history["eval_score"].append(mean_score)
            history["eval_steps"].append(np.mean(steps))
            np.savez(os.path.join(CKPT_DIR, "history"),
                     **{k: np.asarray(v) for k, v in history.items()})
            agent.save(os.path.join(CKPT_DIR, "latest"))
            if mean_score > best_score:
                best_score = mean_score
                agent.save(os.path.join(CKPT_DIR, "best"))
            print(f"步={agent.step_count} eps={agent.eps:.3f} "
                  f"30局均分={mean_score:.1f} 峰值={max(scores)} "
                  f"best={best_score:.1f} 步数={np.mean(steps):.0f} "
                  f"耗时={time.time()-wall:.0f}s")
            next_mark += interval

    agent.save(os.path.join(CKPT_DIR, "final"))
    print("完成，best 30局均分 =", round(best_score, 1))


if __name__ == "__main__":
    train()
