import os, sys

root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame as pg
import launcher

# 菜单 8 个按钮及其处理函数
keys = [b["key"] for b in launcher.BUTTONS]
expected = ["human", "ai", "path", "pretrained", "dqn", "cnn", "world", "beam"]
assert keys == expected, keys
for fn in ["run_human", "run_ai", "run_path", "run_replay", "run_dqn",
           "run_cnn", "run_world", "run_beam"]:
    assert callable(getattr(launcher, fn)), fn

# 无头渲染一次菜单，确认绘制链路无缺失依赖
pg.init()
surf = pg.display.set_mode((launcher.MENU_W, launcher.MENU_H))
launcher.draw_menu(surf, 0)
print("launcher 无头验证通过：8 按钮 =", keys)

# 验证 run_beam 用的离线 agent 与环境能跑
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent
agent = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
agent.load(os.path.join("cnn_ckpt", "best.npz"))
env = SnakeEnv(state_mode="grid", deadline_growth=1.0)
s = env.reset(1)
for _ in range(15):
    a = agent.act(s, training=False, env=env)
    s, _, done, info = env.step(a)
    if done:
        break
print("离线 beam 15 步正常，当前分=", info["score"])
pg.quit()
