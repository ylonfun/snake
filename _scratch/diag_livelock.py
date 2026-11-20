import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import random

from settings import DIRECTIONS, COLS, ROWS
from path_game import SafeShortestSnake

BOARD = [(x, y) for x in range(COLS) for y in range(ROWS)]


def place_food(body, rnd):
    free = [p for p in BOARD if p not in set(body)]
    return rnd.choice(free) if free else None


rnd = random.Random(0)
head = rnd.choice(BOARD)
direction = DIRECTIONS[rnd.randint(0, 3)]
snake = SafeShortestSnake(head, direction, COLS, ROWS)
food = place_food(snake.body, rnd)
checks = {500, 1000, 1500, 1999}

while not snake.dead and food is not None and snake.steps < 2000:
    if snake.move(food):
        food = place_food(snake.body, rnd)
        if food is None:
            break
    if snake.steps in checks:
        h = snake.body[0]
        plain = snake.bfs(h, food, set(snake.body[:-1]))   # 纯 BFS 物理可达
        if plain is None:
            print(f"步{snake.steps} 长度{len(snake.body)} 食物{food}: 纯BFS不可达(物理困住)")
        else:
            safe = snake.path_sustainably_safe(plain)
            print(f"步{snake.steps} 长度{len(snake.body)} 食物{food}: "
                  f"纯BFS长度{len(plain)-1} 持续安全={safe}")
