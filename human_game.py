"""贪吃蛇手动模式：现代深色视觉，方向键转向，R 重开，Esc 退出。"""
import math
import random

import pygame as pg

import theme as T
from theme import CELL, TOP_BAR, COLS, ROWS, GAME_W, GAME_H
from settings import DIRECTIONS


class Game:
    def __init__(self):
        pg.init()
        self.width, self.height = GAME_W, GAME_H
        pg.display.set_caption("贪吃蛇 · 手动模式")
        self.screen = pg.display.set_mode((self.width, self.height))
        self.clock = pg.time.Clock()

        self.score = 0
        self.snake = []
        self.food = None
        self.direction = None
        self.available = {}
        self.game_over = False
        self.exit = False

        self.move_interval = 1000 // 12
        self.last_move = 0

    def play(self):
        self._new()
        self.last_move = pg.time.get_ticks()
        while not self.exit:
            self.clock.tick(60)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.exit = True
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.exit = True
                    elif event.key == pg.K_r and self.game_over:
                        self._new()
                        self.last_move = pg.time.get_ticks()

            keys = pg.key.get_pressed()
            self._handle_keys(keys)

            now = pg.time.get_ticks()
            if not self.game_over and now - self.last_move >= self.move_interval:
                self.last_move = now
                self._move()

            self._draw()
        pg.quit()

    def _handle_keys(self, keys):
        if keys[pg.K_UP] and self.direction != (0, 1):
            self.direction = (0, -1)
        elif keys[pg.K_DOWN] and self.direction != (0, -1):
            self.direction = (0, 1)
        elif keys[pg.K_LEFT] and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif keys[pg.K_RIGHT] and self.direction != (-1, 0):
            self.direction = (1, 0)

    def _new(self):
        self.game_over = False
        self.snake = []
        self.score = 0
        self.available = {}
        for i in range(COLS):
            for j in range(ROWS):
                self.available[(i, j)] = 1

        col = random.randint(2, COLS - 3)
        row = random.randint(2, ROWS - 3)
        self.head = (col, row)
        self.direction = DIRECTIONS[random.randint(0, 3)]
        b1 = (self.head[0] - self.direction[0], self.head[1] - self.direction[1])
        b2 = (b1[0] - self.direction[0], b1[1] - self.direction[1])
        self.snake = [self.head, b1, b2]
        for p in (self.head, b1, b2):
            self.available.pop(p)
        self._place_food()

    def _place_food(self):
        if not self.available:
            self.game_over = True
            return
        self.food = random.choice(list(self.available.keys()))
        self.available.pop(self.food)

    def _move(self):
        self.head = (self.head[0] + self.direction[0],
                     self.head[1] + self.direction[1])
        self.snake.insert(0, self.head)

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            tail = self.snake.pop()
            self.available[tail] = 1
            if self.head not in self.available:
                self.game_over = True
            else:
                self.available.pop(self.head)

    def _cell_xy(self, pos):
        return pos[0] * CELL, pos[1] * CELL + TOP_BAR

    def _draw_board(self):
        for r in range(ROWS):
            for c in range(COLS):
                color = T.CELL_A if (r + c) % 2 == 0 else T.CELL_B
                pg.draw.rect(self.screen, color,
                             (c * CELL, r * CELL + TOP_BAR, CELL, CELL))

    def _draw_snake(self):
        n = len(self.snake)
        for i in range(n - 1, -1, -1):
            x, y = self._cell_xy(self.snake[i])
            t = i / max(1, n - 1)
            color = tuple(int(T.SNAKE_HEAD[k] + (T.SNAKE_TAIL[k] - T.SNAKE_HEAD[k]) * t)
                          for k in range(3))
            T.rounded_rect(self.screen, pg.Rect(x + 2, y + 2, CELL - 4, CELL - 4),
                           color, 9)

        hx, hy = self._cell_xy(self.snake[0])
        dx, dy = self.direction
        if dx != 0:
            ex = hx + CELL // 2 + dx * CELL // 5
            eyes = [(ex, hy + CELL // 3), (ex, hy + 2 * CELL // 3)]
        else:
            ey = hy + CELL // 2 + dy * CELL // 5
            eyes = [(hx + CELL // 3, ey), (hx + 2 * CELL // 3, ey)]
        for ex, ey in eyes:
            pg.draw.circle(self.screen, T.EYE, (ex, ey), 4)
            pg.draw.circle(self.screen, T.TEXT, (ex, ey), 2)

    def _draw_food(self):
        x, y = self._cell_xy(self.food)
        cx, cy = x + CELL // 2, y + CELL // 2
        pulse = math.sin(pg.time.get_ticks() / 260)
        r = int(CELL // 3 + pulse * 2)
        pg.draw.circle(self.screen, T.FOOD_DEEP, (cx, cy), r + 2)
        pg.draw.circle(self.screen, T.FOOD, (cx, cy), r)
        pg.draw.circle(self.screen, T.FOOD_CORE, (cx - r // 3, cy - r // 3),
                       max(2, r // 3))

    def _draw_top_bar(self):
        pg.draw.rect(self.screen, T.BG_BOTTOM, (0, 0, GAME_W, TOP_BAR))
        pg.draw.line(self.screen, T.BORDER_SOFT, (0, TOP_BAR - 1),
                     (GAME_W, TOP_BAR - 1))
        l = T.font(15, bold=True).render("手动模式", True, T.TEAL_BRIGHT)
        self.screen.blit(l, (18, 12))
        d = T.font(14).render(f"得分 {self.score}", True, T.TEXT_DIM)
        self.screen.blit(d, (18, 34))
        h = T.font(13).render("Esc 退出 · R 重开", True, T.TEXT_FAINT)
        self.screen.blit(h, h.get_rect(right=GAME_W - 18, centery=TOP_BAR // 2))

    def _draw_game_over(self):
        overlay = pg.Surface((GAME_W, GAME_H), pg.SRCALPHA)
        overlay.fill((9, 11, 19, 212))
        self.screen.blit(overlay, (0, 0))
        crect = pg.Rect(0, 0, 308, 220)
        crect.center = (GAME_W // 2, GAME_H // 2)
        T.card(self.screen, crect, T.SURFACE, 18, T.BORDER, 1)
        t = T.font(24, bold=True).render("游戏结束", True, T.TEXT)
        self.screen.blit(t, t.get_rect(center=(crect.centerx, crect.y + 44)))
        s = T.font(16).render(f"本局得分：{self.score}", True, T.TEAL_BRIGHT)
        self.screen.blit(s, s.get_rect(center=(crect.centerx, crect.y + 96)))
        h = T.font(14).render("R：重新开始    Esc：返回菜单", True, T.TEXT_DIM)
        self.screen.blit(h, h.get_rect(center=(crect.centerx, crect.y + 140)))

    def _draw(self):
        self._draw_board()
        if self.food is not None:
            self._draw_food()
        self._draw_snake()
        self._draw_top_bar()
        if self.game_over:
            self._draw_game_over()
        pg.display.flip()


if __name__ == "__main__":
    Game().play()
