"""标准 RL 环境（Gym 风格）：纯逻辑、不依赖 pygame。

封装贪吃蛇为 reset/step 接口，产出标准 transition (s, a, r, s', done)。
状态编码与 ai_game.Snake.get_state 完全一致（32 维），移动/碰撞/计分规则
也与 ai_game 保持一致，以便 DQN 与未来世界模型共用同一环境。
"""
import random

import numpy as np

from settings import DIRECTIONS, ROWS, COLS

# 8 个射线方向（与 ai_game.get_state 相同的顺序）
_VISION_DIRS = [[0, -1], [1, -1], [1, 0], [1, 1],
                 [0, 1], [-1, 1], [-1, 0], [-1, -1]]


class SnakeEnv:
    def __init__(self, rows=ROWS, cols=COLS, step_limit=4000, state_mode="local",
                 reward_food=10.0, reward_death=-10.0, reward_step=-0.02,
                 energy_mode=False, allow_stay=True, e_init=2.0, e_gain=1.0,
                 c_move=0.05, c_idle=0.01, survival_mode=False, r_alive=0.1,
                 r_surv_food=1.0, init_len=1, min_len=1, repeat_penalty=0.05,
                 food_deadline=60, deadline_growth=0.0, scent_base=0.015,
                 scent_falloff=0.75):
        self.rows = rows
        self.cols = cols
        self.step_limit = step_limit
        self.state_mode = state_mode
        self.energy_mode = energy_mode
        self.survival_mode = survival_mode
        if energy_mode:
            self.state_dim = 43          # 40 维 local + 计数1 + 气味1 + 能量1
            self.n_actions = 5 if allow_stay else 4
        elif survival_mode:
            self.state_dim = 43          # 40 维 local + 计数1 + 气味1 + 长度1
            self.n_actions = 5           # 4 移动 + shrink
        else:
            self.grid_channels = 6       # grid 模式：头/身/食物/气味/访问/边界
            if state_mode == "grid":
                self.state_dim = self.grid_channels * rows * cols
            else:
                self.state_dim = 300 if state_mode == "full" else 42
            self.n_actions = 4
        self.allow_stay = allow_stay
        self.r_alive = r_alive
        self.r_surv_food = r_surv_food
        self.init_len, self.min_len = init_len, min_len
        self.r_food = reward_food
        self.r_death = reward_death
        self.r_step = reward_step
        self.repeat_penalty = repeat_penalty
        self.visit_count = np.zeros((rows, cols), dtype=np.int64)
        self.food_deadline = food_deadline
        # 动态饥饿期限：每比初始长度多长一格，期限增加 deadline_growth 步。
        self.deadline_growth = deadline_growth
        self.steps_since_food = 0
        self.scent_base = scent_base        # 苹果格最大气味，严格 < |r_step|
        self.scent_falloff = scent_falloff  # 每远离苹果一格的指数衰减率
        self.scent = np.zeros((rows, cols), dtype=np.float32)
        self.death_cause = None
        self.e_init, self.e_gain = e_init, e_gain
        self.c_move, self.c_idle = c_move, c_idle
        self.board = [(x, y) for x in range(cols) for y in range(rows)]
        # 苹果只在内部格生成（排除最外圈边界），避免食物贴边
        self.food_cells = [(x, y) for (x, y) in self.board
                          if 0 < x < cols - 1 and 0 < y < rows - 1]
        self.rand = random.Random()

    def reset(self, seed=None):
        if seed is not None:
            self.rand = random.Random(seed)
        self.body = self._initial_body()
        self.food = self._place_food()
        self.steps = 0
        self.done = False
        self.death_cause = None
        self.energy = self.e_init
        self.steps_since_food = 0
        self.visit_count = np.zeros((self.rows, self.cols), dtype=np.int64)
        for (bx, by) in self.body:
            self.visit_count[by, bx] += 1
        self._build_scent()
        return self.get_state()

    def _build_scent(self):
        """全局气味：以苹果为中心按曼哈顿距离指数衰减，不截断、任何格子都有方向感。"""
        self.scent = np.zeros((self.rows, self.cols), dtype=np.float32)
        fx, fy = self.food
        for y in range(self.rows):
            for x in range(self.cols):
                d = abs(x - fx) + abs(y - fy)
                self.scent[y, x] = self.scent_base * self.scent_falloff ** d

    def _initial_body(self):
        """用蛇形扫描顺序可靠生成 init_len 节的有序身体；反转后 body[0] 为头。"""
        order = []
        for y in range(self.rows):
            xs = range(self.cols) if y % 2 == 0 else range(self.cols - 1, -1, -1)
            for x in xs:
                order.append((x, y))
        cells = order[:min(self.init_len, len(order))]
        body = cells[::-1]
        if len(body) > 1:
            self.direction = (body[0][0] - body[1][0], body[0][1] - body[1][1])
        else:
            self.direction = DIRECTIONS[self.rand.randint(0, 3)]
        return body

    def step(self, action):
        if self.energy_mode:
            return self._step_energy(action)
        if self.survival_mode:
            return self._step_survival(action)
        new_direction = DIRECTIONS[action]
        # 禁止 180 度直接反向：反向输入被忽略，保持当前方向直行（标准贪吃蛇规则）。
        if (new_direction[0] == -self.direction[0]
                and new_direction[1] == -self.direction[1]):
            new_direction = self.direction
        self.direction = new_direction
        old_head = self.body[0]
        head = (old_head[0] + self.direction[0],
                old_head[1] + self.direction[1])

        # 撞墙或撞自身（尾巴本步会移走，故排除 body[-1]）。
        hit_wall = (head[0] < 0 or head[0] >= self.cols
                    or head[1] < 0 or head[1] >= self.rows)
        hit_self = head in self.body[:-1]
        if hit_wall or hit_self:
            self.done = True
            self.death_cause = "wall" if hit_wall else "self"
            return self.get_state(), self.r_death, True, self._info()

        self.body.insert(0, head)
        self.visit_count[head[1], head[0]] += 1
        n_visits = self.visit_count[head[1], head[0]]
        if head == self.food:
            reward = self.r_food
            self.steps_since_food = 0          # 吃到苹果，饥饿计数清零
            free = self._free_cells()
            if not free:
                # 吃满整个棋盘，自然结束。
                self.done = True
                self.death_cause = "full"
                self.steps += 1
                return self.get_state(), reward, True, self._info(full=True)
            self.food = self.rand.choice(free)
            self._build_scent()                # 新苹果出现，重建气味场
        else:
            self.body.pop()
            # 气味奖励：读取当前格静态气味（气味场在本苹果周期内不衰减）
            scent_here = self.scent[head[1], head[0]]
            # 基础步成本 - 重复访问惩罚 + 气味奖励
            reward = (self.r_step - self.repeat_penalty * max(0, n_visits - 1)
                      + scent_here)
            self.steps_since_food += 1

        self.steps += 1
        # 饥饿死亡：动态期限随体长增长（长身绕路需要更多时间），默认 growth=0 即固定。
        deadline = self.food_deadline
        if self.deadline_growth:
            deadline = self.food_deadline + self.deadline_growth * (
                len(self.body) - 1)
        if deadline and self.steps_since_food >= deadline:
            self.done = True
            self.death_cause = "hunger"
        if self.steps >= self.step_limit:
            self.done = True
        return self.get_state(), reward, self.done, self._info()

    def _step_energy(self, action):
        """能量模式：奖励=ΔE；移动/静止消耗代谢，吃苹果补能，能量归零饿死。"""
        self.steps += 1
        delta = 0.0

        if action == 4:                                  # stay：身体完全不动
            self.energy -= self.c_idle
            delta -= self.c_idle
        else:
            self.direction = DIRECTIONS[action]
            old_head = self.body[0]
            head = (old_head[0] + self.direction[0],
                    old_head[1] + self.direction[1])
            # 碰撞：终止（未来收益归零即最大惩罚），本步奖励 0
            if (head[0] < 0 or head[0] >= self.cols
                    or head[1] < 0 or head[1] >= self.rows
                    or head in self.body[:-1]):
                self.done = True
                return self.get_state(), 0.0, True, self._info()

            self.body.insert(0, head)
            self.energy -= self.c_move
            delta -= self.c_move
            if head == self.food:
                self.energy += self.e_gain
                delta += self.e_gain
                free = self._free_cells()
                if not free:
                    self.done = True
                    return self.get_state(), delta, True, self._info(full=True)
                self.food = self.rand.choice(free)
            else:
                self.body.pop()

        if self.energy <= 0:                             # 饿死
            self.done = True
        if self.steps >= self.step_limit:
            self.done = True
        return self.get_state(), delta, self.done, self._info()

    def _step_survival(self, action):
        """生存模式：目标是寿命。shrink 缩身腾空间，吃苹果补储备，长度归零即死。"""
        self.steps += 1
        reward = self.r_alive

        if action == 4:                                  # shrink：原地移除尾部
            if len(self.body) <= 1:
                self.done = True                         # 把自己吃光
                return self.get_state(), 0.0, True, self._info()
            if len(self.body) > self.min_len:
                self.body.pop()
            # 已到 min_len 下限：保持不缩短，本步仍计存活奖励
        else:
            self.direction = DIRECTIONS[action]
            old_head = self.body[0]
            head = (old_head[0] + self.direction[0],
                    old_head[1] + self.direction[1])
            # 碰撞：终止
            if (head[0] < 0 or head[0] >= self.cols
                    or head[1] < 0 or head[1] >= self.rows
                    or head in self.body[:-1]):
                self.done = True
                return self.get_state(), 0.0, True, self._info()
            self.body.insert(0, head)
            reward = self.r_alive
            if head == self.food:
                reward += self.r_surv_food
                free = self._free_cells()
                if not free:
                    self.done = True
                    return self.get_state(), reward, True, self._info(full=True)
                self.food = self.rand.choice(free)
            else:
                self.body.pop()

        if self.steps >= self.step_limit:
            self.done = True
        return self.get_state(), reward, self.done, self._info()

    def get_state(self):
        """局部特征：40 维射线特征 + 访问计数 + 气味（full/grid 走空间表征）。"""
        if self.state_mode == "grid":
            return self._grid_state()
        if self.state_mode == "full":
            return self._full_state()
        i = DIRECTIONS.index(self.direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        if len(self.body) == 1:
            tail_direction = self.direction
        else:
            tail_direction = (self.body[-2][0] - self.body[-1][0],
                              self.body[-2][1] - self.body[-1][1])
        i = DIRECTIONS.index(tail_direction)
        tail_dir = [0.0, 0.0, 0.0, 0.0]
        tail_dir[i] = 1.0
        state = head_dir + tail_dir

        body_set = set(self.body)
        hard_set = set(self.body[:-1])   # 本步不会移走的硬身体（尾巴会让路）
        hx, hy = self.body[0]
        for d in _VISION_DIRS:
            x, y = hx + d[0], hy + d[1]
            dis = 1.0
            see_food = 0.0
            see_self = 0.0
            while 0 <= x < self.cols and 0 <= y < self.rows:
                if (x, y) == self.food:
                    see_food = 1.0
                elif (x, y) in body_set:
                    see_self = 1.0
                dis += 1
                x += d[0]
                y += d[1]
            # 紧邻身体：该方向第 1 格是否为不会移走的硬身体（撞上去即死）
            adj_self = 1.0 if (hx + d[0], hy + d[1]) in hard_set else 0.0
            state += [1.0 / dis, see_food, see_self, adj_self]
        base = np.asarray(state, dtype=np.float32)
        if self.state_mode != "full":
            # 当前头所在格子的重复程度：第1次0.1、重复越多越趋近1，让智能体"看见"自己在重复
            n_vis = self.visit_count[self.body[0][1], self.body[0][0]]
            base = np.append(base, np.clip(n_vis / 10.0, 0, 1))
            # 当前头格子的气味浓度，让智能体"闻得见"苹果方向，世界模型才能在想象里利用气味梯度
            base = np.append(base, self.scent[self.body[0][1], self.body[0][0]])
        if self.energy_mode:
            base = np.append(base, np.clip(self.energy / self.e_init, 0, 5))
        elif self.survival_mode:
            base = np.append(base, len(self.body) / (self.cols * self.rows))
        return base.astype(np.float32)

    def _full_state(self):
        """完整地图：100 格 × 3 通道（头/身体/食物），flatten 成 300 维。"""
        head_ch = np.zeros(self.cols * self.rows, np.float32)
        body_ch = np.zeros(self.cols * self.rows, np.float32)
        food_ch = np.zeros(self.cols * self.rows, np.float32)
        head_ch[self.body[0][1] * self.cols + self.body[0][0]] = 1
        for p in self.body:
            body_ch[p[1] * self.cols + p[0]] = 1
        food_ch[self.food[1] * self.cols + self.food[0]] = 1
        return np.concatenate([head_ch, body_ch, food_ch])

    def _grid_state(self):
        """完整空间表征：(C, H, W) 多通道网格，保持空间结构、局面唯一可确定。

        通道顺序：头、身体、食物、气味、访问计数、边界。前三个为 0/1，气味为
        连续浓度（携带全局梯度），访问为重复程度，边界帮助识别墙。
        """
        head = np.zeros((self.rows, self.cols), np.float32)
        body = np.zeros((self.rows, self.cols), np.float32)
        food = np.zeros((self.rows, self.cols), np.float32)
        border = np.zeros((self.rows, self.cols), np.float32)
        hx, hy = self.body[0]
        head[hy, hx] = 1.0
        for (bx, by) in self.body:
            body[by, bx] = 1.0
        food[self.food[1], self.food[0]] = 1.0
        border[0, :] = 1.0
        border[-1, :] = 1.0
        border[:, 0] = 1.0
        border[:, -1] = 1.0
        visit = np.clip(self.visit_count / 10.0, 0, 1).astype(np.float32)
        return np.stack([head, body, food, self.scent, visit, border], axis=0)

    def _free_cells(self):
        """可放苹果的空格：优先内部格（不贴边），内部全满时才回退到边界格。"""
        occupied = set(self.body)
        free = [p for p in self.food_cells if p not in occupied]
        return free if free else [p for p in self.board if p not in occupied]

    def _place_food(self):
        return self.rand.choice(self._free_cells())

    def clone_state(self):
        """深拷贝内部状态，用于在副本里做想象 rollout，不影响真实环境。

        气味场是苹果位置的确定性函数，恢复时重建，故不单独保存。
        """
        return {
            "body": list(self.body),
            "food": self.food,
            "direction": self.direction,
            "visit_count": self.visit_count.copy(),
            "steps_since_food": self.steps_since_food,
            "steps": self.steps,
            "done": self.done,
            "death_cause": self.death_cause,
            "rand": self.rand.getstate(),
            "energy": self.energy,
        }

    def restore_state(self, snap):
        """恢复 clone_state 保存的快照，并重建气味场。"""
        self.body = list(snap["body"])
        self.food = snap["food"]
        self.direction = snap["direction"]
        self.visit_count = snap["visit_count"].copy()
        self.steps_since_food = snap["steps_since_food"]
        self.steps = snap["steps"]
        self.done = snap["done"]
        self.death_cause = snap["death_cause"]
        if "rand" in snap:
            self.rand.setstate(snap["rand"])
        if "energy" in snap:                    # 旧快照无该字段时保持构造值
            self.energy = snap["energy"]
        self._build_scent()

    def _info(self, full=False):
        return {"score": len(self.body) - 1, "steps": self.steps, "full": full}
