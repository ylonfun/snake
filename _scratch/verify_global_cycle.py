"""global_cycle 多尺寸验证：构造合法性 + HamiltonSnake 吃满。"""
import os, sys, random
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from settings import DIRECTIONS
from path_game import (HamiltonSnake, global_cycle, validate_cycle,
                       validate_partial_cycle)

for W in (10, 11, 12, 14, 15):
    H = W
    cycle, excluded = global_cycle(W, H)
    if excluded is None:
        validate_cycle(cycle, W, H)
    else:
        validate_partial_cycle(cycle, W, H, excluded)
    target = len(cycle) - 1                      # 初始1格，吃满=格数-1

    rnd = random.Random(8000 + W)
    head = rnd.choice(cycle); direction = DIRECTIONS[rnd.randint(0, 3)]
    snake = HamiltonSnake(head, direction, W, H, cycle=cycle, use_shortcuts=True)
    food = rnd.choice([p for p in cycle if p not in set(snake.body)])
    while not snake.dead:
        if snake.move(food):
            free = [p for p in cycle if p not in set(snake.body)]
            if not free:
                break
            food = rnd.choice(free)
    print(f"{W}x{H} 回路{len(cycle)}格 排除={excluded} 吃{snake.score}/"
          f"目标{target} 死亡={snake.dead}")
