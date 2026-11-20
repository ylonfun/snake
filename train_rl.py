"""统一训练器：驱动 env + agent，记录学习曲线，保存可续训 checkpoint，
并周期性用纯贪心策略（ε=0）做闭环评估。训练器只依赖 BaseAgent 接口，
未来世界模型可复用同一训练流程。
"""
import os
import time

import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent

CKPT_DIR = "dqn_ckpt"


def build_env(state_mode="local", **kwargs):
    """按模式构建环境：energy 走能量代谢，其余沿用 local/full。"""
    if state_mode == "energy":
        return SnakeEnv(energy_mode=True, **kwargs)
    if state_mode == "energy_move":
        return SnakeEnv(energy_mode=True, allow_stay=False, **kwargs)
    if state_mode == "survival":
        return SnakeEnv(survival_mode=True, **kwargs)
    return SnakeEnv(state_mode=state_mode, **kwargs)


def ckpt_dir_for(state_mode):
    if state_mode == "full":
        return CKPT_DIR + "_full"
    if state_mode == "energy":
        return CKPT_DIR + "_energy"
    if state_mode == "energy_move":
        return CKPT_DIR + "_energy_move"
    if state_mode == "survival":
        return CKPT_DIR + "_survival"
    return CKPT_DIR


def save_checkpoint(agent, path):
    data = {"step": agent.step_count}
    for i in range(len(agent.q.W)):
        data[f"W{i}"] = agent.q.W[i]
        data[f"b{i}"] = agent.q.b[i]
    np.savez(path, **data)


def load_checkpoint(agent, path):
    data = np.load(path)
    for i in range(len(agent.q.W)):
        agent.q.W[i] = data[f"W{i}"].copy()
        agent.q.b[i] = data[f"b{i}"].copy()
    agent.step_count = int(data["step"])
    agent.sync_target(hard=True)


def evaluate(agent, n_games=10, base_seed=5000, state_mode="local"):
    scores, steps_list = [], []
    for seed in range(base_seed, base_seed + n_games):
        env = build_env(state_mode)
        state = env.reset(seed)
        while True:
            action = agent.act(state, training=False)   # 纯贪心，不探索
            state, _, done, info = env.step(action)
            if done:
                scores.append(info["score"])
                steps_list.append(info["steps"])
                break
    return scores, steps_list


def train(total_steps=200000, interval=20000, resume=None, state_mode="local"):
    ckpt_dir = ckpt_dir_for(state_mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    env = build_env(state_mode)
    agent = DQNAgent(state_mode=state_mode, eps_decay_steps=80000)
    if resume:
        load_checkpoint(agent, resume)

    state = env.reset()
    ep_rewards, ep_scores, cur_r = [], [], 0.0
    history = {"step": [], "train_score": [], "eval_score": [], "eval_steps": []}
    next_mark = max(interval, agent.step_count + interval)
    wall = time.time()

    while agent.step_count < total_steps:
        action = agent.act(state)
        s2, r, done, info = env.step(action)
        agent.remember(state, action, r, s2, done)
        agent.learn()
        cur_r += r
        state = s2
        if done:
            ep_rewards.append(cur_r)
            ep_scores.append(info["score"])
            cur_r = 0.0
            state = env.reset()

        if agent.step_count >= next_mark:
            scores, steps = evaluate(agent, state_mode=state_mode)
            history["step"].append(agent.step_count)
            history["train_score"].append(np.mean(ep_scores[-20:]) if ep_scores else 0)
            history["eval_score"].append(np.mean(scores))
            history["eval_steps"].append(np.mean(steps))
            np.savez(os.path.join(ckpt_dir, "history"),
                     **{k: np.asarray(v) for k, v in history.items()})
            save_checkpoint(agent, os.path.join(ckpt_dir, "latest"))
            print(f"步={agent.step_count} eps={agent.eps:.3f} "
                  f"训练近20局={history['train_score'][-1]:.1f} "
                  f"评估均分={np.mean(scores):.1f} 评估步数={np.mean(steps):.0f} "
                  f"耗时={time.time()-wall:.0f}s")
            next_mark += interval

    save_checkpoint(agent, os.path.join(ckpt_dir, "final"))
    return agent


if __name__ == "__main__":
    train()
