"""15×15 去角 Hamilton 回路驱动 HamiltonSnake 完整局验证（3 局）。"""
import os, sys, json, random, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
from settings import DIRECTIONS
from path_game import HamiltonSnake

with open(os.path.join("_scratch", "cycle15.json")) as f:
    cycle = [tuple(q) for q in json.load(f)]
BOARD = cycle                                   # 蛇只能在这 224 格活动


def place_food(body, rnd):
    free = [p for p in BOARD if p not in set(body)]
    return rnd.choice(free) if free else None


for seed in range(6000, 6003):
    rnd = random.Random(seed)
    head = rnd.choice(BOARD)
    direction = DIRECTIONS[rnd.randint(0, 3)]
    snake = HamiltonSnake(head, direction, 15, 15, cycle=cycle,
                          use_shortcuts=True)
    food = place_food(snake.body, rnd); t0 = time.time()
    while not snake.dead and food is not None:
        if snake.move(food):
            food = place_food(snake.body, rnd)
            if food is None:
                break
    print(f"seed{seed} 吃{snake.score} 步数{snake.steps} 死亡={snake.dead} "
          f"耗时={time.time()-t0:.1f}s")
