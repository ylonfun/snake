import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import random

import numpy as np

from settings import DIRECTIONS, COLS, ROWS
from ai_game import Snake as GASnake
from path_game import HamiltonSnake
from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from dreamer_agent import DreamerAgent
from train_rl import load_checkpoint, CKPT_DIR

N_GAMES = 30
BOARD = [(x, y) for x in range(COLS) for y in range(ROWS)]


def place_food(body, rnd):
    free = [p for p in BOARD if p not in set(body)]
    return rnd.choice(free) if free else None


def init_rnd(seed):
    rnd = random.Random(seed)
    head = rnd.choice(BOARD)
    direction = DIRECTIONS[rnd.randint(0, 3)]
    return rnd, head, direction


def rollout_ga(genes, seed):
    rnd, head, direction = init_rnd(seed)
    snake = GASnake(head, direction, genes.copy(), COLS, ROWS)
    food = place_food(snake.body, rnd)
    while not snake.dead and food is not None:
        if snake.move(food):
            food = place_food(snake.body, rnd)
            if food is None:
                break
    return snake.score, snake.steps


def rollout_ham(seed):
    rnd, head, direction = init_rnd(seed)
    snake = HamiltonSnake(head, direction, COLS, ROWS, use_shortcuts=True)
    food = place_food(snake.body, rnd)
    while not snake.dead and food is not None:
        if snake.move(food):
            food = place_food(snake.body, rnd)
            if food is None:
                break
    return snake.score, snake.steps


def _make_env(rnd, head, direction):
    """按 init_rnd 的初始局面手工装配 env，补齐 reset() 会做的初始化。"""
    env = SnakeEnv()
    env.rand = rnd
    env.body, env.direction = [head], direction
    env.steps, env.done = 0, False
    env.food = place_food(env.body, rnd)
    env.visit_count[head[1], head[0]] += 1    # 与 reset() 一致的初始访问标记
    env._build_scent()                         # 气味场随初始苹果重建
    return env


def rollout_dqn(agent, seed):
    rnd, head, direction = init_rnd(seed)
    env = _make_env(rnd, head, direction)
    state = env.get_state()
    while True:
        action = agent.act(state[:40], training=False)
        state, _, done, info = env.step(action)
        if done:
            return info["score"], info["steps"]


def rollout_dreamer(agent, seed):
    rnd, head, direction = init_rnd(seed)
    env = _make_env(rnd, head, direction)
    state = env.get_state()
    while True:
        action = agent.act(state, training=False)
        state, _, done, info = env.step(action)
        if done:
            return info["score"], info["steps"]


def summarize(name, results):
    scores = np.asarray([r[0] for r in results])
    steps = np.asarray([r[1] for r in results])
    print(f"{name:14s} 均分={scores.mean():6.1f} 平均步数={steps.mean():7.0f} "
          f"最高={scores.max():2d} 到99={int((scores==99).sum()):2d}局")


with open(os.path.join("genes", "record", "genes")) as f:
    ga_genes = np.array(list(map(float, f.read().split())))
dqn = DQNAgent()
load_checkpoint(dqn, os.path.join(CKPT_DIR, "final.npz"))

dreamer = DreamerAgent(freeze_world=True)
dreamer.load_world("world_frozen.npz")
dreamer.load(os.path.join("dreamer_ckpt", "latest.npz"))

summarize("遗传算法GA", [rollout_ga(ga_genes, s) for s in range(N_GAMES)])
summarize("DQN", [rollout_dqn(dqn, s) for s in range(N_GAMES)])
summarize("世界模型Dreamer", [rollout_dreamer(dreamer, s) for s in range(N_GAMES)])
summarize("哈密顿", [rollout_ham(s) for s in range(N_GAMES)])
