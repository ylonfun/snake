import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
import pygame as pg

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint
from settings import (DIRECTIONS, GRID_SIZE, BLANK_SIZE, ROWS, COLS,
                      BLACK, WHITE1, RED, BLUE1, YELLOW)

pg.init()
agent = DQNAgent(state_mode="local")
load_checkpoint(agent, "dqn_ckpt/final.npz")


def safe_actions(e):
    body_set = set(e.body[:-1])
    hx, hy = e.body[0]
    safe = []
    for k, (dx, dy) in enumerate(DIRECTIONS):
        nx, ny = hx + dx, hy + dy
        if (0 <= nx < e.cols and 0 <= ny < e.rows and (nx, ny) not in body_set):
            safe.append(k)
    return safe


def is_typical(e, a, safe):
    hx, hy = e.body[0]
    edge = hx in (0, e.cols - 1) or hy in (0, e.rows - 1)
    if len(e.body) > 1:
        neck = (e.body[1][0] - e.body[0][0], e.body[1][1] - e.body[0][1])
        turning = DIRECTIONS[a] != neck
    else:
        turning = True
    return edge and turning and len(safe) >= 1


captured = []
for seed in range(5000, 5080):
    if len(captured) >= 2:
        break
    e = SnakeEnv(state_mode="local")
    st = e.reset(seed)
    while True:
        safe = safe_actions(e)
        hx, hy = e.body[0]
        a = agent.act(st, training=False)
        dx, dy = DIRECTIONS[a]
        hit = (hx + dx, hy + dy)
        st, r, done, info = e.step(a)
        if done:
            if is_typical(e, a, safe) and hit in set(e.body):
                captured.append((seed, list(e.body), e.food, (hx, hy), hit, len(safe)))
            break


def render(idx, body, food, head, hit, n_safe):
    W, H = COLS * GRID_SIZE, ROWS * GRID_SIZE + BLANK_SIZE
    surf = pg.Surface((W, H))
    surf.fill(BLACK)
    get_xy = lambda p: (p[0] * GRID_SIZE, p[1] * GRID_SIZE + BLANK_SIZE)
    for i in range(COLS + 1):
        pg.draw.line(surf, BLUE1, (i * GRID_SIZE, BLANK_SIZE), (i * GRID_SIZE, H), 1)
    for j in range(ROWS + 1):
        pg.draw.line(surf, BLUE1, (0, j * GRID_SIZE + BLANK_SIZE),
                     (W, j * GRID_SIZE + BLANK_SIZE), 1)
    fx, fy = get_xy(food)
    pg.draw.rect(surf, RED, pg.Rect(fx, fy, GRID_SIZE, GRID_SIZE))
    for k, p in enumerate(body):
        x, y = get_xy(p)
        if k == 0:
            pg.draw.rect(surf, WHITE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        else:
            pg.draw.rect(surf, (0, 180, 0), pg.Rect(x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
    hx, hy = get_xy(hit)
    pg.draw.rect(surf, YELLOW, pg.Rect(hx, hy, GRID_SIZE, GRID_SIZE), 3)
    cx, cy = get_xy(head)
    pg.draw.line(surf, YELLOW, (cx + GRID_SIZE // 2, cy + GRID_SIZE // 2),
                 (hx + GRID_SIZE // 2, hy + GRID_SIZE // 2), 3)
    font = pg.font.Font(None, 26)
    t = font.render(f"Death {idx}: yellow box = collision cell, safe moves left = {n_safe}",
                    True, (255, 255, 255))
    surf.blit(t, (8, 8))
    out = os.path.join(os.getcwd(), f"typical_death_{idx}.png")
    pg.image.save(surf, out)
    print("saved", out, "seed=", seed)


for i, (seed, body, food, head, hit, n_safe) in enumerate(captured, 1):
    render(i, body, food, head, hit, n_safe)
