import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
from types import SimpleNamespace
import pygame as pg

import launcher as L
from dqn_agent import DQNAgent
from rl_env import SnakeEnv
from train_rl import load_checkpoint, CKPT_DIR

pg.init()
surf = pg.display.set_mode((L.GAME_W, L.GAME_H))

# 1) DQN 渲染管线：加载真实模型，跑 30 步并完整渲染
agent = DQNAgent()
load_checkpoint(agent, os.path.join(CKPT_DIR, "final.npz"))
env = SnakeEnv()
state = env.reset(0)
for _ in range(30):
    action = agent.act(state, training=False)
    state, _, done, info = env.step(action)
    view = SimpleNamespace(body=env.body, direction=env.direction,
                           score=info["score"], steps=info["steps"])
    L.draw_board(surf)
    L.draw_food(surf, env.food)
    L.draw_snake(surf, view)
    L.draw_top_bar(surf, "DQN", view.score, -1, steps=view.steps)
    pg.display.flip()
    if done:
        break
print("DQN 渲染管线 30 步无异常")

# 2) 世界模型占位：注入按键让提示框正常退出
pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN))
L.run_world()
print("世界模型占位入口正常")
print("M5 运行逻辑无头验证通过")
