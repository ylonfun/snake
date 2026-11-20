import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import random

import numpy as np

from settings import DIRECTIONS, COLS, ROWS
from path_game import SafeShortestSnake

BOARD = [(x, y) for x in range(COLS) for y in range(ROWS)]
STEP_CAP = 2000


def place_food(body, rnd):
    free = [p for p in BOARD if p not in set(body)]
    return rnd.choice(free) if free else None


for seed in range(10):
    rnd = random.Random(seed)
    head = rnd.choice(BOARD)
    direction = DIRECTIONS[rnd.randint(0, 3)]
    snake = SafeShortestSnake(head, direction, COLS, ROWS)
    food = place_food(snake.body, rnd)
    capped = False
    while not snake.dead and food is not None:
        if snake.move(food):
            food = place_food(snake.body, rnd)
            if food is None:
                break
        if snake.steps >= STEP_CAP:
            capped = True
            break
    print(f"seed={seed} 分数={snake.score} 步数={snake.steps} "
          f"死亡={snake.dead} 超循环={capped}")
