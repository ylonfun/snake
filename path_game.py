"""基于广度优先搜索（BFS）的贪吃蛇最短路径控制器。

与神经网络 + 遗传算法不同，这里不做任何学习或训练：每一步都用 BFS
在棋盘上直接算出最短路径，因此决策是即时的、确定的。

策略（贪心 + 安全检查）：
1. BFS 找一条到食物的最短路径；
2. 用一条"虚拟蛇"模拟沿该路径吃完食物，检查吃完后蛇头是否还能追到
   自己的尾巴（即是否还有逃生空间），安全才走；
3. 吃食物不安全时，改为追自己的尾巴，绕圈保命、等待空间；
4. 极端情况下走向任何能存活的邻居。
"""
import time
from collections import deque

from settings import DIRECTIONS


class PathSnake:
    """一条由 BFS 最短路径驱动的蛇，接口与 ai_game.Snake 对齐。"""

    def __init__(self, head, direction, cols, rows):
        self.body = [head]
        self.direction = direction
        self.cols = cols
        self.rows = rows
        self.score = 0
        self.steps = 0
        self.dead = False

    # ---------- 基础工具 ----------

    def neighbors(self, pos):
        """棋盘内的相邻格子及对应方向。"""
        x, y = pos
        for d in DIRECTIONS:
            nxt = (x + d[0], y + d[1])
            if 0 <= nxt[0] < self.cols and 0 <= nxt[1] < self.rows:
                yield nxt, d

    @staticmethod
    def direction_to(head, nxt):
        return (nxt[0] - head[0], nxt[1] - head[1])

    def bfs(self, start, goal, blocked):
        """BFS 求 start 到 goal 的最短路径。

        blocked 中的格子视为障碍，但 goal 本身始终允许到达
        （用于"追尾巴"，而尾巴当前还在身体里）。
        返回格子序列（含两端），不可达时返回 None。
        """
        if start == goal:
            return [start]
        queue = deque([start])
        came = {start: None}
        while queue:
            cur = queue.popleft()
            for nxt, _ in self.neighbors(cur):
                if nxt in came:
                    continue
                if nxt != goal and nxt in blocked:
                    continue
                came[nxt] = cur
                if nxt == goal:
                    path = [nxt]
                    while came[nxt] is not None:
                        nxt = came[nxt]
                        path.append(nxt)
                    path.reverse()
                    return path
                queue.append(nxt)
        return None

    def simulate_to_food(self, path):
        """复制当前蛇，沿到食物的 path 走一遍，返回吃完食物后的身体。

        中途若发生碰撞返回 None。最后一步吃食物、尾巴不移走（变长）。
        """
        body = list(self.body)
        last = len(path) - 1
        for i in range(1, len(path)):
            new_head = path[i]
            if new_head in body[:-1]:  # 尾巴下一步会移走，排除
                return None
            body.insert(0, new_head)
            if i < last:
                body.pop()
        return body

    # ---------- 决策 ----------

    def choose_direction(self, food):
        head = self.body[0]
        blocked = set(self.body[:-1])  # 尾巴会移走，不算障碍

        # 1 & 2. 找食物并验证吃完后能否追到尾巴（逃生空间）
        food_path = self.bfs(head, food, blocked)
        if food_path is not None:
            future_body = self.simulate_to_food(food_path)
            if future_body is not None:
                future_head = future_body[0]
                future_tail = future_body[-1]
                escape = self.bfs(future_head, future_tail,
                                  set(future_body[:-1]))
                if escape is not None:
                    return self.direction_to(head, food_path[1])

        # 3. 吃食物不安全：追尾巴保命
        tail = self.body[-1]
        tail_path = self.bfs(head, tail, blocked)
        if tail_path is not None and len(tail_path) > 1:
            return self.direction_to(head, tail_path[1])

        # 4. 紧急：走向任何能存活的邻居
        for nxt, d in self.neighbors(head):
            if nxt not in self.body[:-1]:
                return d
        return self.direction

    # ---------- 实际移动 ----------

    def move(self, food):
        """按 BFS 决策移动一步，返回是否吃到食物。"""
        self.steps += 1
        self.direction = self.choose_direction(food)
        head = self.body[0]
        new_head = (head[0] + self.direction[0],
                    head[1] + self.direction[1])

        has_eat = False
        if (new_head[0] < 0 or new_head[0] >= self.cols
                or new_head[1] < 0 or new_head[1] >= self.rows
                or new_head in self.body[:-1]):
            self.dead = True
            return has_eat

        self.body.insert(0, new_head)
        if new_head == food:
            self.score += 1
            has_eat = True
            return has_eat
        self.body.pop()
        return has_eat


# ---------- 哈密顿回路：确定性满分策略 ----------

def hamilton_cycle(width, height):
    """构造 width×height 网格的哈密顿回路（要求 width 为偶数）。

    第一列作为"下行通道"，中间区域按列蛇形覆盖，第一行作为"返回通道"。
    返回格子序列：长度 = width*height，每格恰好一次，首尾相邻。
    """
    path = [(0, 0)]
    for y in range(1, height):              # 第一列向下
        path.append((0, y))
    for c in range(1, width):               # 列 1..width-1 蛇形（行 1..height-1）
        if c % 2 == 1:
            ys = range(height - 1, 0, -1)   # 奇数列：自下而上
        else:
            ys = range(1, height)           # 偶数列：自上而下
        for y in ys:
            path.append((c, y))
    path.append((width - 1, 0))             # 上到右上角
    for c in range(width - 2, 0, -1):       # 第一行向左返回
        path.append((c, 0))
    return path


def validate_cycle(path, width, height):
    """断言这是一条合法哈密顿回路。"""
    assert len(path) == width * height, "长度必须等于格子总数"
    assert len(set(path)) == width * height, "每个格子恰好出现一次"
    for a, b in zip(path, path[1:] + path[:1]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "相邻格必须曼哈顿距离为 1"
    return True


def hamilton_cycle_corner(width, height, exclude=(0, 0)):
    """去掉一个角格后的哈密顿回路（适用于 width 为奇数的网格）。

    返回 (cycle, excluded)，cycle 长度 = width*height-1，结构与偶数版同构：
    左上角局部、第一行右行、末列下行、蛇形列、最左双列锯齿收尾。
    """
    ex, ey = exclude
    p = [(ex, ey + 1), (ex + 1, ey + 1), (ex + 1, ey)]
    p += [(x, ey) for x in range(ex + 2, width)]
    p += [(width - 1, y) for y in range(ey + 1, height)]
    for k, c in enumerate(range(width - 2, ex + 1, -1)):
        if k % 2 == 0:
            p += [(c, y) for y in range(height - 1, ey, -1)]
        else:
            p += [(c, y) for y in range(ey + 1, height)]
    p += [(ex + 1, height - 1), (ex, height - 1)]
    for y in range(height - 2, ey + 1, -1):
        if y % 2 == 1:
            p += [(ex, y), (ex + 1, y)]
        else:
            p += [(ex + 1, y), (ex, y)]
    return p, exclude


def global_cycle(width, height):
    """任意尺寸的可吃满全局回路。

    返回 (cycle, excluded)：width 偶数时全格回路、excluded=None；
    width 奇数时去掉一个角格、excluded 为该格（牺牲 1 格换确定吃满）。
    """
    if width % 2 == 0:
        return hamilton_cycle(width, height), None
    return hamilton_cycle_corner(width, height)


def validate_partial_cycle(path, width, height, excluded):
    """断言去角回路合法（覆盖除 excluded 外的全部格、首尾相邻）。"""
    n = width * height - 1
    assert len(path) == n and len(set(path)) == n
    assert excluded not in path
    for a, b in zip(path, path[1:] + path[:1]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    return True


class HamiltonSnake:
    """沿固定哈密顿回路移动的蛇，数学上保证不撞死、最终满分。

    两级捷径（use_shortcuts=True）：
    - 多步 BFS（主）：在"只允许回路索引前进"的子图上直接搜索到食物的
      最短路径。由于索引严格递增，身体始终维持 尾->身->头 的顺序、不会
      乱序，因此既大幅缩短绕路，又保证安全。身体占比达到 cutoff 后停用。
    - 单步跳跃（兜底）：跳到回路更靠前的相邻格，要求回路弧 (head, target]
      内不含身体段，保证跳跃不会越过身体、破坏回路结构。
    """

    def __init__(self, head, direction, cols, rows, cycle=None,
                 use_shortcuts=True, cutoff=0.55):
        self.cols, self.rows = cols, rows
        self.cycle = cycle or hamilton_cycle(cols, rows)
        self.n = len(self.cycle)
        self.index = {pos: i for i, pos in enumerate(self.cycle)}
        self.body = [head]
        self.direction = direction
        self.score = 0
        self.steps = 0
        self.dead = False
        self.use_shortcuts = use_shortcuts
        # 身体占比达到该值后禁用捷径、纯走回路（空间紧张时求稳）。
        # 0.55 为 0.2~1.0 网格扫描后的实验最优值（50 局平均步数最低、仍 100% 满分）。
        self.cutoff = cutoff
        self.plan_time = 0.0

    def _grid_neighbors(self, pos):
        x, y = pos
        for d in DIRECTIONS:
            nxt = (x + d[0], y + d[1])
            if 0 <= nxt[0] < self.cols and 0 <= nxt[1] < self.rows:
                yield nxt, d

    def _choose(self, food):
        h = self.index[self.body[0]]
        normal = self.cycle[(h + 1) % self.n]
        if not self.use_shortcuts or len(self.body) >= self.n:
            return normal
        # 身体占比达到 cutoff：纯走回路求稳（边界用 >=，与 cutoff 对齐）
        if len(self.body) / self.n >= self.cutoff:
            return normal
        # 空间充足：多步 BFS 直连食物，最短路径
        path = self._search(food)
        if path is not None and len(path) >= 2:
            return path[1]
        # 兜底：单步捷径
        return self._choose_single(food)

    def _search(self, food):
        """在"只允许回路索引前进"的子图上 BFS 搜到食物的最短路径。

        索引严格递增保证无环、节点不超过 n；身体维持回路顺序、不会乱序。
        返回格子列表（含起点、食物），找不到返回 None。
        """
        t0 = time.perf_counter()
        n = self.n
        h0 = self.index[self.body[0]]
        tail_idx = self.index[self.body[-1]]
        food_idx = self.index[food]
        hard = (tail_idx - h0) % n - 2          # 头到尾的安全前进空间
        fd = (food_idx - h0) % n                # 头到食物的前进距离
        limit = min(hard, fd)
        path = None

        if limit >= 1:
            blocked = set(self.body[:-1])        # 尾巴会移走，不算障碍
            start = self.body[0]
            came = {start: None}
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                h = self.index[cur]
                for nxt, _ in self._grid_neighbors(cur):
                    if nxt not in self.index:
                        continue
                    if nxt in came:
                        continue
                    d = (self.index[nxt] - h) % n
                    total = (self.index[nxt] - h0) % n
                    if d < 1 or total > limit or nxt in blocked:
                        continue
                    came[nxt] = cur
                    if nxt == food:
                        p = [nxt]
                        while came[nxt] is not None:
                            nxt = came[nxt]
                            p.append(nxt)
                        path = p[::-1]
                        queue.clear()
                        break
                    queue.append(nxt)

        self.plan_time += time.perf_counter() - t0
        return path

    def _choose_single(self, food):
        h = self.index[self.body[0]]
        normal = self.cycle[(h + 1) % self.n]
        if not self.use_shortcuts or len(self.body) >= self.n:
            return normal

        L = len(self.body)
        d_food = (self.index[food] - h) % self.n       # 前进方向头到食物的距离
        full_idx = set(self.index[p] for p in self.body)
        tail_idx = self.index[self.body[-1]]
        best_d, best_pos = 1, normal                  # 兜底：正常前进一步

        for nxt, _ in self._grid_neighbors(self.body[0]):
            if nxt not in self.index:
                continue
            eats = (nxt == food)
            body_idx = set(full_idx)
            if not eats:
                body_idx.discard(tail_idx)            # 不吃食物时尾巴这一步会移走
            j = self.index[nxt]
            d = (j - h) % self.n
            # 必须前进、不越过食物、与尾部留空间（同时排除反向 d=n-1）、目标空闲
            if d < 1 or d > d_food or d >= self.n - L or j in body_idx:
                continue
            # 回路弧 (h, j] 内不含身体
            ok, k = True, (h + 1) % self.n
            while k != j:
                if k in body_idx:
                    ok = False
                    break
                k = (k + 1) % self.n
            if ok and d > best_d:
                best_d, best_pos = d, nxt
        return best_pos

    def move(self, food):
        """移动一步（可能抄近路），返回是否吃到食物。"""
        self.steps += 1
        nxt_pos = self._choose(food)
        self.direction = PathSnake.direction_to(self.body[0], nxt_pos)

        has_eat = False
        if nxt_pos in self.body[:-1]:           # 防御性检查（理论上不会发生）
            self.dead = True
            return has_eat
        self.body.insert(0, nxt_pos)
        if nxt_pos == food:
            self.score += 1
            has_eat = True
            return has_eat
        self.body.pop()
        return has_eat
