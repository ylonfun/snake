"""体检补充：GA 存档往返、手动模式逻辑与真实渲染出图（无头）。"""
import os
import sys
import tempfile

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame as pg

pg.init()

# ---- 1) GA save_checkpoint / load_checkpoint 往返 ----
from main import GA

ga = GA(p_size=4, c_size=2)
ga.generate_ancestor()
ga.evolve()
tmp = tempfile.mkdtemp(prefix="ga_ckpt_")
ga.save_checkpoint(tmp)
ga2 = GA(p_size=4, c_size=2)
ck = ga2.load_checkpoint(tmp)
assert ck is not None, "存档往返失败"
record, gen = ck if isinstance(ck, tuple) and len(ck) >= 2 else (None, None)
assert len(ga2.population) == 4 and len(ga2.population[0].genes) == 964
print(f"[PASS] GA 存档往返：代={ga2.generation} 种群={len(ga2.population)} 返回={ck}")

# ---- 2) 手动模式逻辑（无键盘输入，直接驱动 _move）----
import random
import human_game

pg.display.set_mode((human_game.GAME_W, human_game.GAME_H))
h = human_game.Game()
h._new()
sc0 = h.score
for _ in range(30):
    if h.game_over:
        break
    # 每步挑一个不会立即撞墙/撞身的合法方向
    cands = []
    for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx = h.head[0] + d[0]
        ny = h.head[1] + d[1]
        if not (0 <= nx < human_game.COLS and 0 <= ny < human_game.ROWS):
            continue
        if d == (-h.direction[0], -h.direction[1]):
            continue
        if (nx, ny) not in h.available and (nx, ny) != h.food:
            continue
        cands.append(d)
    if cands:
        h.direction = random.choice(cands)
    h._move()
h._draw()
pg.image.save(h.screen, os.path.join("_scratch", "health_human.png"))
print(f"[PASS] 手动模式 30 步：得分 {sc0}->{h.score} 结束={h.game_over} 蛇长={len(h.snake)}")

# ---- 3) 菜单与对局画面渲染出图 ----
import launcher

surf = pg.display.set_mode((launcher.MENU_W, launcher.MENU_H))
launcher.draw_menu(surf, hover=None)
pg.image.save(surf, os.path.join("_scratch", "health_menu.png"))

g = pg.display.set_mode((launcher.GAME_W, launcher.GAME_H))
from rl_env import SnakeEnv
from types import SimpleNamespace
env = SnakeEnv()
state = env.reset(1234)
for _ in range(12):
    env.step(int(np.random.randint(4)))
view = SimpleNamespace(body=env.body, direction=env.direction,
                       score=len(env.body) - 3, steps=env.steps)
launcher.draw_board(g)
launcher.draw_food(g, env.food)
launcher.draw_snake(g, view)
launcher.draw_top_bar(g, "功能体检", len(env.body) - 3, -1, steps=env.steps)
pg.image.save(g, os.path.join("_scratch", "health_board.png"))
print("[PASS] 菜单/对局画面已渲染并保存 PNG")
print("OK")
