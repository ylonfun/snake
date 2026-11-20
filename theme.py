"""贪吃蛇 AI 的现代视觉主题：统一配色、尺寸与可复用绘制组件。"""
from functools import lru_cache

import pygame as pg

# ---------- 逻辑网格 ----------
COLS, ROWS = 10, 10

# ---------- 显示尺寸 ----------
CELL = 36
TOP_BAR = 62
BOARD_W = CELL * COLS
BOARD_H = CELL * ROWS
GAME_W = BOARD_W
GAME_H = BOARD_H + TOP_BAR
MENU_W, MENU_H = 400, 710
RADIUS = 14

# ---------- 背景 ----------
BG_TOP = (27, 30, 43)
BG_BOTTOM = (15, 17, 26)
GLOW_TEAL = (45, 212, 191)
GLOW_BLUE = (56, 130, 246)

# ---------- 表面 / 卡片 ----------
SURFACE = (33, 39, 56)
SURFACE_HOVER = (45, 54, 78)
SURFACE_ACTIVE = (52, 64, 92)
BORDER = (70, 80, 108)
BORDER_SOFT = (48, 56, 78)

# ---------- 品牌色 ----------
TEAL = (45, 212, 191)
TEAL_BRIGHT = (94, 234, 212)
TEAL_DARK = (20, 140, 130)
BLUE = (56, 130, 246)
BLUE_BRIGHT = (96, 165, 250)
PURPLE = (139, 92, 246)
AMBER = (245, 158, 11)
AMBER_BRIGHT = (251, 191, 36)
GREEN = (52, 199, 89)
GREEN_DARK = (28, 120, 60)
CYAN = (34, 211, 238)
CYAN_DARK = (18, 110, 130)

# ---------- 文字 ----------
TEXT = (237, 241, 248)
TEXT_DIM = (158, 170, 190)
TEXT_FAINT = (105, 116, 138)

# ---------- 棋盘 ----------
CELL_A = (25, 29, 43)
CELL_B = (29, 34, 50)
GRID_LINE = (38, 44, 62)

# ---------- 蛇 ----------
SNAKE_HEAD = (94, 234, 212)
SNAKE_BODY = (45, 212, 191)
SNAKE_TAIL = (24, 125, 118)
EYE = (13, 17, 25)

# ---------- 食物 ----------
FOOD = (251, 113, 133)
FOOD_CORE = (253, 164, 175)
FOOD_DEEP = (244, 63, 94)


# ---------- 字体 ----------

@lru_cache(maxsize=32)
def font(size, bold=False):
    return pg.font.SysFont("microsoftyahei,simhei,arial", size, bold=bold)


# ---------- 基础形状 ----------

def rounded_rect(surf, rect, color, radius=RADIUS, width=0):
    pg.draw.rect(surf, color, rect, width, border_radius=radius)


def card(surf, rect, fill, radius=RADIUS, border=None, border_w=1):
    if border is not None:
        rounded_rect(surf, rect, border, radius)
        inner = rect.inflate(-border_w * 2, -border_w * 2)
        rounded_rect(surf, inner, fill, max(2, radius - border_w))
    else:
        rounded_rect(surf, rect, fill, radius)


def vertical_gradient(surf, top_color, bottom_color, rect=None):
    if rect is None:
        x, y, w, h = 0, 0, surf.get_width(), surf.get_height()
    else:
        x, y, w, h = rect
    for i in range(h):
        t = i / max(1, h - 1)
        c = tuple(int(top_color[k] + (bottom_color[k] - top_color[k]) * t)
                  for k in range(3))
        pg.draw.line(surf, c, (x, y + i), (x + w, y + i))


def radial_glow(surf, center, radius, color, max_alpha=70):
    for r in range(radius, 0, -4):
        a = int(max_alpha * (1 - r / radius))
        glow = pg.Surface((r * 2, r * 2), pg.SRCALPHA)
        pg.draw.circle(glow, (*color, a), (r, r), r)
        surf.blit(glow, (center[0] - r, center[1] - r))


# ---------- 图标（线性几何，白色描边） ----------

def draw_dpad_icon(surf, cx, cy, size, color=TEXT):
    arm = size // 2
    t = max(3, size // 6)
    pg.draw.rect(surf, color, (cx - t // 2, cy - arm, t, arm * 2),
                 border_radius=2)
    pg.draw.rect(surf, color, (cx - arm, cy - t // 2, arm * 2, t),
                 border_radius=2)
    pg.draw.circle(surf, color, (cx, cy), max(2, t // 2))


def draw_network_icon(surf, cx, cy, size, color=TEXT):
    r = max(3, size // 9)
    left = [(cx - size // 2, cy - size // 3),
            (cx - size // 2, cy + size // 3)]
    mid = [(cx, cy - size // 4), (cx, cy + size // 4)]
    right = [(cx + size // 2, cy)]
    for a in left:
        for b in mid:
            pg.draw.line(surf, color, a, b, 2)
    for a in mid:
        for b in right:
            pg.draw.line(surf, color, a, b, 2)
    for p in left + mid + right:
        pg.draw.circle(surf, color, p, r)


def draw_trophy_icon(surf, cx, cy, size, color=AMBER_BRIGHT):
    w = size // 2
    top = cy - size // 2
    pg.draw.rect(surf, color, (cx - w // 2, top, w, size // 2),
                 border_radius=4)
    pg.draw.arc(surf, color, (cx - w // 2 - w // 3, top, w // 2, size // 3),
                -1.57, 1.57, 3)
    pg.draw.arc(surf, color, (cx + w // 2, top, w // 2, size // 3),
                1.57, 4.71, 3)
    pg.draw.rect(surf, color, (cx - 2, top + size // 2, 4, size // 5))
    pg.draw.rect(surf, color, (cx - w // 2, top + size // 2 + size // 6,
                               w, size // 7), border_radius=3)


def draw_pause_glyph(surf, rect, color=TEXT):
    w = max(3, rect.width // 5)
    gap = rect.width // 4
    y0 = rect.centery - rect.height // 3
    h = rect.height * 2 // 3
    pg.draw.rect(surf, color, (rect.centerx - gap - w // 2, y0, w, h),
                 border_radius=2)
    pg.draw.rect(surf, color, (rect.centerx + gap - w // 2, y0, w, h),
                 border_radius=2)


def draw_path_icon(surf, cx, cy, size, color=TEXT):
    """折线路径图标：节点 + 连线，象征最短路径规划。"""
    pts = [(cx - size // 2, cy + size // 3),
           (cx - size // 6, cy - size // 4),
           (cx + size // 2, cy - size // 3)]
    t = max(2, size // 9)
    pg.draw.lines(surf, color, False, pts, t)
    for p in pts:
        pg.draw.circle(surf, color, p, max(2, size // 7))


def draw_target_icon(surf, cx, cy, size, color=TEXT):
    """靶心：同心圆 + 中心点，象征闭环试错逼近目标。"""
    r = size // 2
    pg.draw.circle(surf, color, (cx, cy), r, 2)
    pg.draw.circle(surf, color, (cx, cy), r * 2 // 3, 2)
    pg.draw.circle(surf, color, (cx, cy), max(2, r // 4))


def draw_globe_icon(surf, cx, cy, size, color=TEXT):
    """地球：外圆 + 经纬线，象征内部世界模型。"""
    r = size // 2
    pg.draw.circle(surf, color, (cx, cy), r, 2)
    pg.draw.line(surf, color, (cx - r, cy), (cx + r, cy), 2)
    pg.draw.line(surf, color, (cx, cy - r), (cx, cy + r), 2)
    pg.draw.ellipse(surf, color, (cx - r // 2, cy - r, r, 2 * r), 2)
