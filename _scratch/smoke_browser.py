import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame as pg
import launcher as L

pg.init()
seeds = L._random_seeds()

# 1) 播放主循环：替换结束等待，避免卡住
L._end_wait = lambda surf, clock: "back"
surf = pg.display.set_mode((L.GAME_W, L.GAME_H))
r = L._play_world_game(surf, 0, seeds[0])
print("播放返回:", r)
assert r == "back"

# 2) 选择界面 stage 级：先渲染一帧(空事件)，再点击第一个可用按钮
surf = pg.display.set_mode((L.MENU_W, L.MENU_H))
rects = L._grid_rects(pg.Rect(28, 96, L.MENU_W - 56, 470), 2, 5, 14)
click = pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=rects[0].center)
seq = [(), (click,)]
calls = {"i": 0}
orig_get = pg.event.get
def fake_get(*a, **k):
    i = calls["i"]; calls["i"] += 1
    return seq[min(i, 1)]
pg.event.get = fake_get
out = L._choose(surf, "stage", 0, seeds)
print("stage 选择返回:", out)
pg.event.get = orig_get

# 3) game 级点击第一个局
click = pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=rects[0].center)
calls = {"i": 0}
def fake_get2(*a, **k):
    i = calls["i"]; calls["i"] += 1
    return (click,) if i == 1 else ()
pg.event.get = fake_get2
out2 = L._choose(surf, "game", 0, seeds)
print("game 选择返回:", out2)
pg.event.get = orig_get
print("冒烟通过")
