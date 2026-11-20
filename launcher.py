"""贪吃蛇 AI 启动器：现代深色界面，手动 / AI 进化 / 最佳回放。"""
import math
import os
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pygame as pg

import theme as T
from theme import (
    CELL, TOP_BAR, COLS, ROWS, GAME_W, GAME_H, MENU_W, MENU_H,
)
from human_game import Game as HumanGame
from ai_game import Snake
from path_game import HamiltonSnake
from main import GA
from settings import DIRECTIONS
from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from dreamer_agent import DreamerAgent
from train_rl import load_checkpoint as load_dqn, CKPT_DIR as DQN_DIR

# 演示节奏
FAST_FPS = 60
SLOW_FPS = 25
SLOW_AFTER = 100
PATH_FPS = 120

# 回放速度档位：慢档逐格观察，快档快速看整体
SPEED_GEARS = [
    {"name": "慢", "fps": 8},
    {"name": "中", "fps": 25},
    {"name": "快", "fps": 60},
]
DEFAULT_GEAR = 1

PAUSE_BTN = pg.Rect(GAME_W - 66, 14, 52, 34)
SPEED_BTN = pg.Rect(GAME_W - 150, 14, 76, 34)

LOG_DIR, LOG_FILE = "logs", os.path.join("logs", "evolution.log")
CHECKPOINT_DIR = os.path.join("genes", "myrun")
RECORD_DIR = os.path.join("genes", "record")

BUTTONS = [
    {"key": "human", "title": "手动玩", "sub": "方向键转向，蛇自动前进",
     "icon": "dpad", "tint": T.TEAL, "tint_dark": T.TEAL_DARK},
    {"key": "ai", "title": "AI 自动玩", "sub": "遗传算法连续进化，可暂停",
     "icon": "network", "tint": T.BLUE, "tint_dark": (28, 70, 140)},
    {"key": "path", "title": "路径 AI", "sub": "哈密顿回路，零训练保证满分",
     "icon": "path", "tint": T.PURPLE, "tint_dark": (88, 55, 170)},
    {"key": "pretrained", "title": "最佳回放", "sub": "回放最强蛇，同分选步数最少",
     "icon": "trophy", "tint": T.AMBER, "tint_dark": (150, 95, 10)},
    {"key": "dqn", "title": "DQN 强化学习", "sub": "闭环试错，奖励驱动自主学习",
     "icon": "target", "tint": T.GREEN, "tint_dark": T.GREEN_DARK},
    {"key": "cnn", "title": "CNN 深度规划", "sub": "多步空间前瞻，实时安全决策",
     "icon": "network", "tint": T.TEAL, "tint_dark": T.TEAL_DARK},
    {"key": "world", "title": "世界模型 Dreamer", "sub": "学内部模型，在想象中规划",
     "icon": "globe", "tint": T.CYAN, "tint_dark": T.CYAN_DARK},
    {"key": "beam", "title": "离线束搜索", "sub": "多路径深度规划，逼近吃满",
     "icon": "trophy", "tint": T.AMBER, "tint_dark": (150, 95, 10)},
]


# ---------- 通用工具 ----------

def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def lerp(c1, c2, t):
    return tuple(int(c1[k] + (c2[k] - c1[k]) * t) for k in range(3))


def cell_xy(pos):
    """Snake body 元组 (列 x, 行 y) -> 像素。"""
    return pos[0] * CELL, pos[1] * CELL + TOP_BAR


# ---------- 菜单 ----------

def draw_menu(surf, hover):
    T.vertical_gradient(surf, T.BG_TOP, T.BG_BOTTOM)
    T.radial_glow(surf, (70, 130), 190, T.GLOW_TEAL, 55)
    T.radial_glow(surf, (MENU_W - 50, 320), 210, T.GLOW_BLUE, 50)

    # 品牌区
    logo = pg.Rect(34, 46, 52, 52)
    T.card(surf, logo, T.TEAL_DARK, 14, T.TEAL, 1)
    T.draw_network_icon(surf, logo.centerx, logo.centery, 28, T.TEXT)
    title = T.font(32, bold=True).render("贪吃蛇 AI", True, T.TEXT)
    surf.blit(title, (100, 48))
    sub = T.font(15).render("神经网络 · 遗传算法 · 交互式演示", True, T.TEXT_DIM)
    surf.blit(sub, (102, 92))

    card_w, card_h, gap = 340, 64, 8
    x0 = (MENU_W - card_w) // 2
    y0 = 96
    for i, b in enumerate(BUTTONS):
        rect = pg.Rect(x0, y0 + i * (card_h + gap), card_w, card_h)
        active = i == hover
        fill = T.SURFACE_HOVER if active else T.SURFACE
        border = b["tint"] if active else T.BORDER_SOFT
        T.card(surf, rect, fill, 16, border, 2 if active else 1)

        # 图标圆底
        icx, icy = rect.x + 44, rect.centery
        pg.draw.circle(surf, b["tint_dark"], (icx, icy), 24)
        if active:
            pg.draw.circle(surf, b["tint"], (icx, icy), 24, 2)
        if b["icon"] == "dpad":
            T.draw_dpad_icon(surf, icx, icy, 23)
        elif b["icon"] == "network":
            T.draw_network_icon(surf, icx, icy, 23)
        elif b["icon"] == "path":
            T.draw_path_icon(surf, icx, icy, 23)
        elif b["icon"] == "target":
            T.draw_target_icon(surf, icx, icy, 23)
        elif b["icon"] == "globe":
            T.draw_globe_icon(surf, icx, icy, 23)
        else:
            T.draw_trophy_icon(surf, icx, icy, 23)

        t_sfc = T.font(21, bold=True).render(b["title"], True, T.TEXT)
        surf.blit(t_sfc, (rect.x + 84, rect.y + 12))
        s_sfc = T.font(13).render(b["sub"], True, T.TEXT_DIM)
        surf.blit(s_sfc, (rect.x + 84, rect.y + 38))

    tip = T.font(13).render("点击卡片进入模式 · 关闭窗口退出", True, T.TEXT_FAINT)
    surf.blit(tip, tip.get_rect(center=(MENU_W // 2, MENU_H - 24)))
    pg.display.flip()


def menu_loop(surf):
    clock = pg.time.Clock()
    card_w, card_h, gap = 340, 64, 8
    x0 = (MENU_W - card_w) // 2
    y0 = 96
    while True:
        hover = None
        mp = pg.mouse.get_pos()
        for i in range(len(BUTTONS)):
            rect = pg.Rect(x0, y0 + i * (card_h + gap), card_w, card_h)
            if rect.collidepoint(mp):
                hover = i
        draw_menu(surf, hover)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                raise SystemExit
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                for i in range(len(BUTTONS)):
                    rect = pg.Rect(x0, y0 + i * (card_h + gap), card_w, card_h)
                    if rect.collidepoint(event.pos):
                        return BUTTONS[i]["key"]
        clock.tick(60)


# ---------- 手动模式 ----------

def run_human():
    HumanGame().play()


# ---------- 游戏渲染 ----------

def draw_board(surf):
    for r in range(ROWS):
        for c in range(COLS):
            color = T.CELL_A if (r + c) % 2 == 0 else T.CELL_B
            pg.draw.rect(surf, color, (c * CELL, r * CELL + TOP_BAR, CELL, CELL))


def draw_snake(surf, snake):
    n = len(snake.body)
    for i in range(n - 1, -1, -1):
        x, y = cell_xy(snake.body[i])
        t = i / max(1, n - 1)
        color = lerp(T.SNAKE_HEAD, T.SNAKE_TAIL, t)
        inset = 2
        T.rounded_rect(surf, pg.Rect(x + inset, y + inset,
                                     CELL - inset * 2, CELL - inset * 2),
                        color, 9)

    # 蛇头眼睛
    hx, hy = cell_xy(snake.body[0])
    dx, dy = snake.direction
    if dx != 0:
        ex = hx + CELL // 2 + dx * CELL // 5
        eyes = [(ex, hy + CELL // 3), (ex, hy + 2 * CELL // 3)]
    else:
        ey = hy + CELL // 2 + dy * CELL // 5
        eyes = [(hx + CELL // 3, ey), (hx + 2 * CELL // 3, ey)]
    for ex, ey in eyes:
        pg.draw.circle(surf, T.EYE, (ex, ey), 4)
        pg.draw.circle(surf, T.TEXT, (ex, ey), 2)


def draw_food(surf, food):
    x, y = cell_xy(food)
    cx, cy = x + CELL // 2, y + CELL // 2
    pulse = math.sin(pg.time.get_ticks() / 260)
    r = int(CELL // 3 + pulse * 2)
    pg.draw.circle(surf, T.FOOD_DEEP, (cx, cy), r + 2)
    pg.draw.circle(surf, T.FOOD, (cx, cy), r)
    pg.draw.circle(surf, T.FOOD_CORE, (cx - r // 3, cy - r // 3),
                   max(2, r // 3))


def draw_top_bar(surf, label, score, record, gear=None, steps=None):
    pg.draw.rect(surf, T.BG_BOTTOM, (0, 0, GAME_W, TOP_BAR))
    pg.draw.line(surf, T.BORDER_SOFT, (0, TOP_BAR - 1), (GAME_W, TOP_BAR - 1))

    l_sfc = T.font(15, bold=True).render(label, True, T.TEAL_BRIGHT)
    surf.blit(l_sfc, (18, 12))
    if steps is not None:
        d_sfc = T.font(14).render(
            f"得分 {score}   步数 {steps}", True, T.TEXT_DIM)
    elif record is None or record < 0:
        d_sfc = T.font(14).render(f"得分 {score}", True, T.TEXT_DIM)
    else:
        d_sfc = T.font(14).render(f"得分 {score}   最高 {record}", True, T.TEXT_DIM)
    surf.blit(d_sfc, (18, 34))

    if gear is not None:
        T.card(surf, SPEED_BTN, T.SURFACE, 9, T.BORDER_SOFT, 1)
        g_sfc = T.font(13, bold=True).render(f"速度 {gear['name']}", True, T.TEXT)
        surf.blit(g_sfc, g_sfc.get_rect(center=SPEED_BTN.center))

    T.card(surf, PAUSE_BTN, T.SURFACE, 9, T.BORDER_SOFT, 1)
    T.draw_pause_glyph(surf, PAUSE_BTN, T.TEXT)


def render_game(surf, snake, food, label, record, gear=None, show_steps=False):
    draw_board(surf)
    draw_food(surf, food)
    draw_snake(surf, snake)
    draw_top_bar(surf, label, snake.score, record, gear,
                 snake.steps if show_steps else None)
    pg.display.flip()


# ---------- 暂停 / 覆盖层 ----------

def show_overlay(surf, title, lines):
    overlay = pg.Surface((GAME_W, GAME_H), pg.SRCALPHA)
    overlay.fill((9, 11, 19, 212))
    surf.blit(overlay, (0, 0))
    crect = pg.Rect(0, 0, 308, 230)
    crect.center = (GAME_W // 2, GAME_H // 2)
    T.card(surf, crect, T.SURFACE, 18, T.BORDER, 1)

    t_sfc = T.font(24, bold=True).render(title, True, T.TEXT)
    surf.blit(t_sfc, t_sfc.get_rect(center=(crect.centerx, crect.y + 42)))
    y = crect.y + 92
    for text, color, bold in lines:
        sfc = T.font(15, bold).render(text, True, color)
        surf.blit(sfc, sfc.get_rect(center=(crect.centerx, y)))
        y += 32
    pg.display.flip()


def pause_loop(surf, clock, label, score, record):
    show_overlay(surf, "已暂停", [
        (f"{label}  得分 {score}  最高 {record}", T.TEXT_DIM, False),
        ("P / 空格：继续", T.TEXT, True),
        ("Esc：返回菜单", T.TEXT_DIM, False),
    ])
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return "quit"
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return "quit"
                if event.key in (pg.K_p, pg.K_SPACE):
                    return "resume"
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if PAUSE_BTN.collidepoint(event.pos):
                    return "resume"
        clock.tick(30)


# ---------- 演示一条蛇 ----------

def place_food(snake, rnd):
    board = set((x, y) for x in range(COLS) for y in range(ROWS))
    for body in snake.body:
        board.discard(body)
    if not board:
        return None
    return rnd.choice(list(board))


def run_demo(surf, clock, genes, seed, label, record, fps, gear=None,
             show_steps=False):
    rnd = random.Random(seed)
    board = [(x, y) for x in range(COLS) for y in range(ROWS)]
    head = rnd.choice(board)
    direction = DIRECTIONS[rnd.randint(0, 3)]
    snake = Snake(head, direction, genes.copy(), COLS, ROWS)
    food = place_food(snake, rnd)

    while not snake.dead and food is not None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return "quit"
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return "quit"
                if event.key in (pg.K_p, pg.K_SPACE):
                    if pause_loop(surf, clock, label, snake.score, record) == "quit":
                        return "quit"
                if gear is not None:
                    if event.key in (pg.K_1, pg.K_KP1):
                        gear = 0
                    elif event.key in (pg.K_2, pg.K_KP2):
                        gear = 1
                    elif event.key in (pg.K_3, pg.K_KP3):
                        gear = 2
                    elif event.key in (pg.K_EQUALS, pg.K_KP_PLUS, pg.K_RIGHT):
                        gear = (gear + 1) % len(SPEED_GEARS)
                    elif event.key in (pg.K_MINUS, pg.K_KP_MINUS, pg.K_LEFT):
                        gear = (gear - 1) % len(SPEED_GEARS)
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if gear is not None and SPEED_BTN.collidepoint(event.pos):
                    gear = (gear + 1) % len(SPEED_GEARS)
                elif PAUSE_BTN.collidepoint(event.pos):
                    if pause_loop(surf, clock, label, snake.score, record) == "quit":
                        return "quit"

        if snake.move(food):
            food = place_food(snake, rnd)
            if food is None:
                return "done"
        if gear is not None:
            render_game(surf, snake, food, label, record,
                       gear=SPEED_GEARS[gear], show_steps=show_steps)
            clock.tick(SPEED_GEARS[gear]["fps"])
        else:
            render_game(surf, snake, food, label, record, show_steps=show_steps)
            clock.tick(fps)
    return "done"


# ---------- AI 自动进化 ----------

def draw_waiting(surf, generation, record, current_best):
    T.vertical_gradient(surf, T.BG_TOP, T.BG_BOTTOM,
                        pg.Rect(0, 0, GAME_W, GAME_H))
    t = T.font(22, bold=True).render(f"正在进化第 {generation} 代…", True, T.TEXT)
    surf.blit(t, t.get_rect(center=(GAME_W // 2, GAME_H // 2 - 40)))
    s = T.font(14).render("正在模拟 500 条蛇，请稍候", True, T.TEXT_DIM)
    surf.blit(s, s.get_rect(center=(GAME_W // 2, GAME_H // 2 - 4)))
    info = T.font(15).render(f"历史最高 {record}   上代最佳 {current_best}",
                             True, T.TEAL_BRIGHT)
    surf.blit(info, info.get_rect(center=(GAME_W // 2, GAME_H // 2 + 36)))
    hint = T.font(13).render("P 暂停 · Esc 返回菜单（刷新纪录时自动演示）",
                             True, T.TEXT_FAINT)
    surf.blit(hint, hint.get_rect(center=(GAME_W // 2, GAME_H // 2 + 78)))
    pg.display.flip()


def pump_train_events():
    """Process events between generations; return 'quit'/'pause'/None."""
    for event in pg.event.get():
        if event.type == pg.QUIT:
            return "quit"
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                return "quit"
            if event.key in (pg.K_p, pg.K_SPACE):
                return "pause"
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if PAUSE_BTN.collidepoint(event.pos):
                return "pause"
    return None


def read_stored_record():
    """Return the stored (score, steps), or None if no record exists.

    Legacy records predate steps.txt; treat their steps as infinity so a
    same-score but shorter genome can replace them on the first encounter.
    """
    score_path = os.path.join(RECORD_DIR, "score.txt")
    if not os.path.exists(score_path):
        return None
    with open(score_path) as f:
        stored_score = int(f.read())
    steps_path = os.path.join(RECORD_DIR, "steps.txt")
    stored_steps = float("inf")
    if os.path.exists(steps_path):
        with open(steps_path) as f:
            stored_steps = int(f.read())
    return stored_score, stored_steps


def save_record(ga):
    """Persist the best genome to the replay directory.

    Ranking is lexicographic: a higher score always wins; on an equal score
    the genome that reaches it in fewer steps wins. Returns whether the
    stored record was actually replaced.
    """
    new_score = ga.best_individual.score
    new_steps = ga.best_individual.steps
    stored = read_stored_record()
    if stored is not None:
        stored_score, stored_steps = stored
        if new_score < stored_score:
            return False
        if new_score == stored_score and new_steps >= stored_steps:
            return False
    os.makedirs(RECORD_DIR, exist_ok=True)
    with open(os.path.join(RECORD_DIR, "genes"), "w") as f:
        f.write(" ".join(map(str, ga.best_individual.genes)))
    with open(os.path.join(RECORD_DIR, "seed"), "w") as f:
        f.write(str(ga.best_individual.seed))
    with open(os.path.join(RECORD_DIR, "score.txt"), "w") as f:
        f.write(str(new_score))
    with open(os.path.join(RECORD_DIR, "steps.txt"), "w") as f:
        f.write(str(new_steps))
    return True


def run_ai():
    ga = GA()
    checkpoint = ga.load_checkpoint(CHECKPOINT_DIR)
    if checkpoint is not None:
        record, _ = checkpoint
        log(f"检测到训练存档，从第 {ga.generation} 代继续，历史最高 {record}")
    else:
        # No resume checkpoint: warm-start from the strongest genome on hand
        # instead of crawling up from a random population.
        warm_path = os.path.join(RECORD_DIR, "genes")
        if os.path.exists(warm_path):
            with open(warm_path) as f:
                base_genes = np.array(list(map(float, f.read().split())))
            ga.seed_from_genes(base_genes)
            with open(os.path.join(RECORD_DIR, "score.txt")) as f:
                record = int(f.read())
            log(f"未发现续训存档，从最强成果（{record} 分）热启动")
        else:
            ga.generate_ancestor()
            record = 0
            log("无任何基线，从随机祖先开始训练（网络 32-20-12-4）")

    surf = pg.display.set_mode((GAME_W, GAME_H))
    clock = pg.time.Clock()
    while True:
        current_best = ga.best_individual.score if ga.best_individual else 0
        draw_waiting(surf, ga.generation + 1, record, current_best)
        action = pump_train_events()
        if action == "quit":
            log("训练结束，返回主菜单。")
            return
        if action == "pause":
            if pause_loop(surf, clock, "训练中", current_best, record) == "quit":
                log("训练结束，返回主菜单。")
                return

        ga.evolve()
        is_record = save_record(ga)
        if is_record:
            record = ga.best_individual.score
        ga.save_checkpoint(CHECKPOINT_DIR)
        log(f"generation: {ga.generation}, record: {record}, "
            f"best: {ga.best_individual.score}, avg: {ga.avg_score:.2f}")

        if is_record:
            if run_demo(surf, clock, ga.best_individual.genes,
                        ga.best_individual.seed,
                        f"第 {ga.generation} 代 · 新纪录",
                        record, FAST_FPS) == "quit":
                log("训练结束，返回主菜单。")
                return


def show_notice(surf, title, message):
    """Centered notice card; dismiss on any key/click."""
    overlay = pg.Surface((GAME_W, GAME_H), pg.SRCALPHA)
    overlay.fill((9, 11, 19, 220))
    surf.blit(overlay, (0, 0))
    crect = pg.Rect(0, 0, 308, 180)
    crect.center = (GAME_W // 2, GAME_H // 2)
    T.card(surf, crect, T.SURFACE, 18, T.BORDER, 1)
    t_sfc = T.font(22, bold=True).render(title, True, T.TEXT)
    surf.blit(t_sfc, t_sfc.get_rect(center=(crect.centerx, crect.y + 48)))
    m_sfc = T.font(14).render(message, True, T.TEXT_DIM)
    surf.blit(m_sfc, m_sfc.get_rect(center=(crect.centerx, crect.y + 100)))
    pg.display.flip()
    notice_clock = pg.time.Clock()
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
            if event.type in (pg.KEYDOWN, pg.MOUSEBUTTONDOWN):
                return
        notice_clock.tick(30)


def run_path():
    """哈密顿回路即时演示：零训练，确定吃满、不撞死。"""
    surf = pg.display.set_mode((GAME_W, GAME_H))
    clock = pg.time.Clock()
    rnd = random.Random()
    cells = [(x, y) for x in range(COLS) for y in range(ROWS)]
    head = rnd.choice(cells)
    direction = DIRECTIONS[rnd.randint(0, 3)]
    snake = HamiltonSnake(head, direction, COLS, ROWS, use_shortcuts=True)
    food = place_food(snake, rnd)
    label = "路径 AI · 哈密顿回路"

    while not snake.dead and food is not None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                log("路径 AI 结束。")
                return
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    log("路径 AI 结束。")
                    return
                if event.key in (pg.K_p, pg.K_SPACE):
                    if pause_loop(surf, clock, label, snake.score, -1) == "quit":
                        return
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if PAUSE_BTN.collidepoint(event.pos):
                    if pause_loop(surf, clock, label, snake.score, -1) == "quit":
                        return

        if snake.move(food):
            food = place_food(snake, rnd)
            if food is None:              # 吃满整个棋盘，结束
                break
        render_game(surf, snake, food, label, -1)
        clock.tick(PATH_FPS)

    log(f"路径 AI 完成，得分 {snake.score}。")


def run_replay():
    surf = pg.display.set_mode((GAME_W, GAME_H))
    genes_path = os.path.join(RECORD_DIR, "genes")
    if not os.path.exists(genes_path):
        show_notice(surf, "还没有可回放的存档", "先点「AI 自动玩」训练出成绩")
        return
    with open(genes_path) as f:
        genes = np.array(list(map(float, f.read().split())))
    with open(os.path.join(RECORD_DIR, "seed")) as f:
        seed = int(f.read())
    with open(os.path.join(RECORD_DIR, "score.txt")) as f:
        top_score = int(f.read())
    steps_path = os.path.join(RECORD_DIR, "steps.txt")
    top_steps = None
    if os.path.exists(steps_path):
        with open(steps_path) as f:
            top_steps = int(f.read())
    clock = pg.time.Clock()
    if top_steps is not None:
        log(f"回放最强基因（{top_score} 分、{top_steps} 步），可切换速度档位")
    else:
        log(f"回放最强基因（{top_score} 分），可切换速度档位")
    run_demo(surf, clock, genes, seed, f"最佳回放 · {top_score} 分",
             -1, FAST_FPS, gear=DEFAULT_GEAR, show_steps=True)


def _play_env_loop(surf, label, make_env, decide, fps=PATH_FPS):
    """三个 env 演示共用的"按种子循环对局 + 渲染"循环。

    make_env(seed) -> 已 reset 的 (env, state)；decide(env, state) -> action。
    关闭窗口或按 ESC 返回；fps 为 None 时不限帧率（决策本身即瓶颈）。
    """
    clock = pg.time.Clock()
    for seed in range(100):
        env, state = make_env(seed)
        view = SimpleNamespace(body=env.body, direction=env.direction,
                               score=0, steps=0)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    return
            action = decide(env, state)
            state, _, done, info = env.step(action)
            view.body, view.direction = env.body, env.direction
            view.score, view.steps = info["score"], info["steps"]

            draw_board(surf)
            draw_food(surf, env.food)
            draw_snake(surf, view)
            draw_top_bar(surf, label, view.score, -1, steps=view.steps)
            pg.display.flip()
            if fps is not None:
                clock.tick(fps)
            if done:
                break


def run_dqn():
    """DQN 演示：加载已训练模型，纯贪心策略连续对局并渲染。"""
    ckpt = os.path.join(DQN_DIR, "final.npz")
    surf = pg.display.set_mode((GAME_W, GAME_H))
    if not os.path.exists(ckpt):
        show_notice(surf, "还没有 DQN 模型", "先运行 train_rl.py 完成训练")
        return
    agent = DQNAgent()
    load_dqn(agent, ckpt)

    def make_env(seed):
        env = SnakeEnv()
        return env, env.reset(seed)

    _play_env_loop(surf, "DQN 强化学习", make_env,
                   lambda env, state: agent.act(state[:40], training=False))


def run_cnn():
    """CNN 深度规划演示：多步空间前瞻，实时安全决策（grid 表征）。"""
    from cnn_safe_agent import CNNMultiSpaceAgent
    ckpt = os.path.join("cnn_ckpt", "best.npz")
    surf = pg.display.set_mode((GAME_W, GAME_H))
    if not os.path.exists(ckpt):
        show_notice(surf, "还没有 CNN 模型", "先运行 train_cnn.py 完成训练")
        return
    agent = CNNMultiSpaceAgent()
    agent.load(ckpt)

    def make_env(seed):
        env = SnakeEnv(state_mode="grid")
        return env, env.reset(seed)

    _play_env_loop(surf, "CNN 深度规划", make_env,
                   lambda env, state: agent.act(state, training=False, env=env))


def run_beam():
    """离线束搜索演示：多路径深度规划，决策较慢但接近吃满（grid 表征）。"""
    from cnn_beam_agent import CNNBeamAgent
    ckpt = os.path.join("cnn_ckpt", "best.npz")
    surf = pg.display.set_mode((GAME_W, GAME_H))
    if not os.path.exists(ckpt):
        show_notice(surf, "还没有 CNN 模型", "先运行 train_cnn.py 完成训练")
        return
    agent = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
    agent.load(ckpt)

    def make_env(seed):
        env = SnakeEnv(state_mode="grid", deadline_growth=1.0)
        return env, env.reset(seed)

    _play_env_loop(surf, "离线束搜索（深度规划）", make_env,
                   lambda env, state: agent.act(state, training=False, env=env),
                   fps=None)


N_STAGES, N_GAMES = 10, 10


def _stage_file(stage):
    return os.path.join("dreamer_ckpt", f"stage_{stage}.npz")


def _random_seeds(n=N_GAMES):
    return [random.randint(10000, 99999) for _ in range(n)]


def _grid_rects(area, cols, rows, gap):
    cw = (area.w - gap * (cols - 1)) // cols
    ch = (area.h - gap * (rows - 1)) // rows
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(pg.Rect(area.x + c * (cw + gap),
                               area.y + r * (ch + gap), cw, ch))
    return out


def _choose(surf, level, stage, seeds):
    """两级选择界面。返回 ('stage',i)/('game',j)/'refresh'/'back'/'quit'。"""
    clock = pg.time.Clock()
    while True:
        T.vertical_gradient(surf, T.BG_TOP, T.BG_BOTTOM)
        if level == "stage":
            title, sub = "选择训练阶段", "共 10 个阶段，每 2 万步存档"
        else:
            title, sub = f"阶段 {stage} · 随机局面", "苹果随机位置，检验临场决策"
        ts = T.font(23, bold=True).render(title, True, T.TEXT)
        surf.blit(ts, ts.get_rect(center=(MENU_W // 2, 38)))
        ss = T.font(14).render(sub, True, T.TEXT_DIM)
        surf.blit(ss, ss.get_rect(center=(MENU_W // 2, 68)))

        area = pg.Rect(28, 96, MENU_W - 56, 470)
        rects = _grid_rects(area, 2, 5, 14)
        mp = pg.mouse.get_pos()
        avail = []
        for i, rct in enumerate(rects):
            if level == "stage":
                ok = os.path.exists(_stage_file(i))
                avail.append(ok)
                hover = ok and rct.collidepoint(mp)
                lab = f"阶段 {i}"
                dlab = f"{(i+1)*2} 万步" if ok else "未存档"
                tcol = T.TEXT if ok else T.TEXT_FAINT
            else:
                hover = rct.collidepoint(mp)
                lab, dlab = f"局 {i+1}", f"种子 {seeds[i]}"
                tcol = T.TEXT
            T.card(surf, rct, T.SURFACE_HOVER if hover else T.SURFACE, 14,
                   T.CYAN if hover else T.BORDER_SOFT, 2 if hover else 1)
            l1 = T.font(19, bold=True).render(lab, True, tcol)
            surf.blit(l1, l1.get_rect(center=(rct.centerx, rct.y + 32)))
            l2 = T.font(13).render(dlab, True, T.TEXT_DIM)
            surf.blit(l2, l2.get_rect(center=(rct.centerx, rct.y + 64)))

        func = {}
        if level == "game":
            func = {"refresh": pg.Rect(28, 588, 160, 50),
                    "back": pg.Rect(212, 588, 160, 50)}
            for key, rct in func.items():
                h = rct.collidepoint(mp)
                txt = "换一批" if key == "refresh" else "返回阶段"
                T.card(surf, rct, T.SURFACE_HOVER if h else T.SURFACE, 12,
                       T.BORDER_SOFT, 1)
                f = T.font(17, bold=True).render(txt, True, T.TEXT)
                surf.blit(f, f.get_rect(center=rct.center))
        tip = T.font(13).render("ESC 退出 · 点击卡片进入", True, T.TEXT_FAINT)
        surf.blit(tip, tip.get_rect(center=(MENU_W // 2, 678)))
        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return "quit"
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return "quit"
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                for i, rct in enumerate(rects):
                    if rct.collidepoint(event.pos):
                        if level == "stage":
                            if avail[i]:
                                return ("stage", i)
                        else:
                            return ("game", i)
                for key, rct in func.items():
                    if rct.collidepoint(event.pos):
                        return key
        clock.tick(60)


def _end_wait(surf, clock):
    """对局结束后保持画面，等待按键/点击返回，或退出。"""
    hint = T.font(15, bold=True).render("本局结束 · 点击或按空格返回", True, T.AMBER_BRIGHT)
    surf.blit(hint, hint.get_rect(center=(GAME_W // 2, 16)))
    pg.display.flip()
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return "quit"
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return "quit"
                return "back"
            if event.type == pg.MOUSEBUTTONDOWN:
                return "back"
        clock.tick(30)


def _play_world_game(surf, stage, seed):
    """加载指定阶段 checkpoint，在随机 seed 局面播放；返回 'quit'/'back'。"""
    agent = DreamerAgent(freeze_world=True)
    agent.load_world("world_frozen.npz")
    agent.load(_stage_file(stage))
    env = SnakeEnv()
    state = env.reset(seed)
    clock = pg.time.Clock()
    gear = DEFAULT_GEAR
    view = SimpleNamespace(body=env.body, direction=env.direction, score=0, steps=0)
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return "quit"
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return "back"
                if event.key in (pg.K_p, pg.K_SPACE):
                    if pause_loop(surf, clock, f"阶段{stage}", view.score, -1) == "quit":
                        return "quit"
                if event.key in (pg.K_1, pg.K_KP1):
                    gear = 0
                elif event.key in (pg.K_2, pg.K_KP2):
                    gear = 1
                elif event.key in (pg.K_3, pg.K_KP3):
                    gear = 2
                elif event.key in (pg.K_EQUALS, pg.K_KP_PLUS, pg.K_RIGHT):
                    gear = (gear + 1) % len(SPEED_GEARS)
                elif event.key in (pg.K_MINUS, pg.K_KP_MINUS, pg.K_LEFT):
                    gear = (gear - 1) % len(SPEED_GEARS)
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if SPEED_BTN.collidepoint(event.pos):
                    gear = (gear + 1) % len(SPEED_GEARS)
                elif PAUSE_BTN.collidepoint(event.pos):
                    if pause_loop(surf, clock, f"阶段{stage}", view.score, -1) == "quit":
                        return "quit"
        action = agent.act(state, training=False)
        state, _, done, info = env.step(action)
        view.body, view.direction = env.body, env.direction
        view.score, view.steps = info["score"], info["steps"]
        draw_board(surf)
        draw_food(surf, env.food)
        draw_snake(surf, view)
        draw_top_bar(surf, f"阶段{stage} 随机局", view.score, -1,
                     gear=SPEED_GEARS[gear], steps=view.steps)
        pg.display.flip()
        clock.tick(SPEED_GEARS[gear]["fps"])
        if done:
            return _end_wait(surf, clock)


def run_world():
    """阶段浏览器：选阶段 → 选随机局面 → 播放，可随时返回/换一批。"""
    if not os.path.exists("world_frozen.npz"):
        surf = pg.display.set_mode((MENU_W, MENU_H))
        show_notice(surf, "还没有世界模型", "先运行 build_world.py 完成训练")
        return
    surf = pg.display.set_mode((MENU_W, MENU_H))
    level, stage = "stage", 0
    seeds = _random_seeds()
    while True:
        choice = _choose(surf, level, stage, seeds)
        if choice == "quit":
            return
        if choice == "back":
            level = "stage"
            continue
        if choice == "refresh":
            seeds = _random_seeds()
            continue
        if level == "stage":
            stage = choice[1]
            seeds = _random_seeds()
            level = "game"
            continue
        j = choice[1]
        surf = pg.display.set_mode((GAME_W, GAME_H))
        result = _play_world_game(surf, stage, seeds[j])
        surf = pg.display.set_mode((MENU_W, MENU_H))
        if result == "quit":
            return


def main():
    pg.init()
    surf = pg.display.set_mode((MENU_W, MENU_H))
    while True:
        choice = menu_loop(surf)
        if choice == "human":
            run_human()
        elif choice == "ai":
            run_ai()
        elif choice == "path":
            run_path()
        elif choice == "pretrained":
            run_replay()
        elif choice == "dqn":
            run_dqn()
        elif choice == "cnn":
            run_cnn()
        elif choice == "world":
            run_world()
        elif choice == "beam":
            run_beam()
        surf = pg.display.set_mode((MENU_W, MENU_H))


if __name__ == "__main__":
    main()
