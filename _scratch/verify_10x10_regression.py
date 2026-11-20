"""标准 10×10 Hamilton 回路回归（确认去角改动不影响原有满分）。"""
import os, sys, random
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from settings import DIRECTIONS
from path_game import HamiltonSnake, hamilton_cycle, validate_cycle

W = H = 10
cycle = hamilton_cycle(W, H); validate_cycle(cycle, W, H)
BOARD = cycle
for seed in range(7000, 7003):
    rnd = random.Random(seed)
    head = rnd.choice(BOARD); direction = DIRECTIONS[rnd.randint(0, 3)]
    snake = HamiltonSnake(head, direction, W, H, cycle=cycle, use_shortcuts=True)
    food = rnd.choice([p for p in BOARD if p not in set(snake.body)])
    while not snake.dead:
        if snake.move(food):
            free = [p for p in BOARD if p not in set(snake.body)]
            if not free:
                break
            food = rnd.choice(free)
    print(f"seed{seed} 吃{snake.score} 死亡={snake.dead}")
