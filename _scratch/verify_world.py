import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame as pg
import launcher as L
from rl_env import SnakeEnv

pg.init()
surf = pg.display.set_mode((L.GAME_W, L.GAME_H))
agent = L.DreamerAgent(freeze_world=True)
agent.load_world("world_frozen.npz")
agent.load(os.path.join("dreamer_ckpt", "latest.npz"))
env = SnakeEnv(); state = env.reset(1)
from types import SimpleNamespace
view = SimpleNamespace(body=env.body, direction=env.direction, score=0, steps=0)
for _ in range(60):
    state, _, done, info = env.step(agent.act(state, False))
    view.body, view.direction = env.body, env.direction
    view.score, view.steps = info["score"], info["steps"]
    L.draw_board(surf); L.draw_food(surf, env.food); L.draw_snake(surf, view)
    L.draw_top_bar(surf, "世界模型 Dreamer", view.score, -1, steps=view.steps)
    pg.display.flip()
    if done:
        break
print("无头世界模型闭环+渲染验证通过，步数", view.steps, "分数", view.score)
pg.quit()
