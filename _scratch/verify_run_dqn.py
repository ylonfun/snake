import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame as pg
pg.init()
import launcher

calls = {"n": 0}
def fake_get():
    calls["n"] += 1
    if calls["n"] < 15:
        return []
    return [pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE)]

launcher.pg.event.get = fake_get
launcher.run_dqn()
print("run_dqn 跑了", calls["n"], "次事件循环后正常退出，无报错")
