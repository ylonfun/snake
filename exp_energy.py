"""能量代谢 DQN 综合实验：含普通 DQN 基线 + 多档 c_move，同口径对比。

指标：分数、路径效率（实际/曼哈顿）、最短段占比、最长无吃果段（活锁检测）、
死亡原因。能量组用无 stay 纯移动模式。
"""
import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint

TRAIN_STEPS = 150000
EVAL_GAMES = 30
EVAL_BASE = 5000


def train_one(c_move, total_steps=TRAIN_STEPS):
    env = SnakeEnv(energy_mode=True, allow_stay=False, c_move=c_move)
    agent = DQNAgent(state_mode="energy_move", eps_decay_steps=80000)
    state = env.reset()
    while agent.step_count < total_steps:
        action = agent.act(state)
        s2, r, done, info = env.step(action)
        agent.remember(state, action, r, s2, done)
        agent.learn()
        state = s2
        if done:
            state = env.reset()
    return agent


def run_eval(agent, make_env, n_games=EVAL_GAMES, base_seed=EVAL_BASE):
    scores, causes = [], []
    ratios, exact, max_seg, total_moves = [], 0, 0, 0
    for seed in range(base_seed, base_seed + n_games):
        env = make_env(seed)
        seg_start = env.body[0]
        seg_steps = 0
        while True:
            action = agent.act(env.get_state(), training=False)
            state, r, done, info = env.step(action)
            seg_steps += 1
            total_moves += 1
            if r > 0:
                manh = abs(seg_start[0] - env.body[0][0]) + abs(
                    seg_start[1] - env.body[0][1])
                if manh > 0:
                    ratios.append(seg_steps / manh)
                    if seg_steps == manh:
                        exact += 1
                max_seg = max(max_seg, seg_steps)
                seg_start = env.body[0]
                seg_steps = 0
            if done:
                scores.append(info["score"])
                if info.get("full"):
                    causes.append("吃满")
                elif getattr(env, "energy_mode", False) and env.energy <= 0:
                    causes.append("饿死")
                else:
                    causes.append("撞死")
                break
    return {
        "score": np.mean(scores), "path_ratio": np.mean(ratios),
        "exact_pct": 100 * exact / max(1, len(ratios)),
        "max_seg": max_seg, "starve": causes.count("饿死"),
        "crash": causes.count("撞死"),
    }


def energy_env_factory(c_move):
    def make(seed):
        env = SnakeEnv(energy_mode=True, allow_stay=False, c_move=c_move)
        env.reset(seed)
        return env
    return make


def baseline_env_factory(seed):
    env = SnakeEnv(state_mode="local")
    env.reset(seed)
    return env


if __name__ == "__main__":
    # 普通 DQN 基线
    base_agent = DQNAgent(state_mode="local")
    load_checkpoint(base_agent, "dqn_ckpt/final.npz")
    m = run_eval(base_agent, baseline_env_factory)
    print(f"普通DQN基线: 均分={m['score']:.1f} 路径比={m['path_ratio']:.2f} "
          f"最短段={m['exact_pct']:.0f}% 最长段={m['max_seg']} "
          f"撞死={m['crash']}")
    # 能量代谢各档
    for cm in [0.02, 0.05, 0.06, 0.07, 0.08]:
        agent = train_one(cm)
        m = run_eval(agent, energy_env_factory(cm))
        print(f"能量c_move={cm}: 均分={m['score']:.1f} 路径比={m['path_ratio']:.2f} "
              f"最短段={m['exact_pct']:.0f}% 最长段={m['max_seg']} "
              f"饿死={m['starve']} 撞死={m['crash']}")
