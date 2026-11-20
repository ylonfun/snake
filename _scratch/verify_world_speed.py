import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame as pg
from types import SimpleNamespace
import launcher
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

pg.init()
surf = pg.display.set_mode((launcher.GAME_W, launcher.GAME_H))
agent = DreamerAgent(freeze_world=True)
agent.load_world("world_frozen.npz")
agent.load("dreamer_ckpt/latest.npz")
env = SnakeEnv()
clock = pg.time.Clock()
gear = launcher.DEFAULT_GEAR
state = env.reset(1)
view = SimpleNamespace(body=env.body, direction=env.direction, score=0, steps=0)

# 依次注入按键 1/3/2/+/−，验证档位切换与渲染不报错
synthetic = [pg.K_1, pg.K_3, pg.K_2, pg.K_EQUALS, pg.K_MINUS]
frames = 0
for key in synthetic:
    pg.event.post(pg.event.Event(pg.KEYDOWN, key=key))
    for event in pg.event.get():
        if event.type == pg.KEYDOWN:
            if event.key in (pg.K_1, pg.K_KP1): gear = 0
            elif event.key in (pg.K_2, pg.K_KP2): gear = 1
            elif event.key in (pg.K_3, pg.K_KP3): gear = 2
            elif event.key in (pg.K_EQUALS, pg.K_KP_PLUS, pg.K_RIGHT):
                gear = (gear + 1) % len(launcher.SPEED_GEARS)
            elif event.key in (pg.K_MINUS, pg.K_KP_MINUS, pg.K_LEFT):
                gear = (gear - 1) % len(launcher.SPEED_GEARS)
    state, _, done, info = env.step(agent.act(state, False))
    view.body, view.direction = env.body, env.direction
    view.score, view.steps = info["score"], info["steps"]
    launcher.draw_board(surf); launcher.draw_food(surf, env.food)
    launcher.draw_snake(surf, view)
    launcher.draw_top_bar(surf, "世界模型 Dreamer", view.score, -1,
                          gear=launcher.SPEED_GEARS[gear], steps=view.steps)
    pg.display.flip()
    clock.tick(launcher.SPEED_GEARS[gear]["fps"])
    frames += 1
    print(f"帧{frames} 档位={launcher.SPEED_GEARS[gear]['name']} fps={launcher.SPEED_GEARS[gear]['fps']}")
    if done:
        state = env.reset(2)

print("世界模型速度调节器无头验证通过，共", frames, "帧")
pg.quit()
