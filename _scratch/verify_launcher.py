import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame as pg

import theme as T
import launcher as L

pg.init()
surf = pg.display.set_mode((T.MENU_W, T.MENU_H))
L.draw_menu(surf, None)
print("菜单渲染成功，按钮数 =", len(L.BUTTONS))

card_w, card_h, gap = 340, 84, 12
x0 = (T.MENU_W - card_w) // 2
y0 = 104
for i, b in enumerate(L.BUTTONS):
    bottom = y0 + i * (card_h + gap) + card_h
    assert bottom < T.MENU_H - 24, (b["key"], bottom)
    print(f"  {b['key']:10s} 底边={bottom}")
print("M5 菜单布局无头验证通过")
