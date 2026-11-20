"""14×14 大偶数棋盘 Hamilton 回路验证（3 局）。"""
import os, sys, random, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from settings import DIRECTIONS
from path_game import HamiltonSnake, validate_cycle, hamilton_cycle

W = H = 14
BOARD = [(x, y) for x in range(W) for y in range(H)]
cycle = hamilton_cycle(W, H)
validate_cycle(cycle, W, H)
print("14x14 哈密顿回路构造合法，长度=", len(cycle))


def place_food(body, rnd):
    free = [p for p in BOARD if p not in set(body)]
    return rnd.choice(free) if free else None


for seed in range(5000, 5003):
    rnd = random.Random(seed)
    head = rnd.choice(BOARD)
    direction = DIRECTIONS[rnd.randint(0, 3)]
    snake = HamiltonSnake(head, direction, W, H, cycle=cycle, use_shortcuts=True)
    food = place_food(snake.body, rnd); t0 = time.time()
    while not snake.dead and food is not None:
        if snake.move(food):
            food = place_food(snake.body, rnd)
            if food is None:
                break
    print(f"seed{seed} 吃{snake.score} 步数{snake.steps} 死亡={snake.dead} "
          f"耗时={time.time()-t0:.1f}s")
