import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint
from settings import DIRECTIONS

agent = DQNAgent(state_mode="local")
load_checkpoint(agent, "dqn_ckpt/final.npz")

# 统计项
safe_count = {0: 0, 1: 0, 2: 0, 3: 0}     # 死亡前还有几个安全动作
head_pos = {"角落": 0, "靠边": 0, "内部": 0}
hit_part = {"靠近头(前1/3)": 0, "中段": 0, "靠近尾(后1/3)": 0}
turn_type = {"直行": 0, "转弯": 0, "掉头": 0}
n = 0

for seed in range(5000, 5040):
    e = SnakeEnv(state_mode="local")
    st = e.reset(seed)
    while True:
        # 动作前：计算各方向是否安全
        body_set = set(e.body[:-1])
        hx, hy = e.body[0]
        safe = []
        for k, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = hx + dx, hy + dy
            if (0 <= nx < e.cols and 0 <= ny < e.rows and (nx, ny) not in body_set):
                safe.append(k)
        a = agent.act(st, training=False)
        # 颈部方向（body[0] -> body[1]）
        if len(e.body) > 1:
            neck = (e.body[1][0]-e.body[0][0], e.body[1][1]-e.body[0][1])
        else:
            neck = None
        st, r, done, info = e.step(a)
        if done:
            n += 1
            safe_count[len(safe)] += 1
            # 头位置
            edge_x = hx in (0, e.cols-1)
            edge_y = hy in (0, e.rows-1)
            if edge_x and edge_y:
                head_pos["角落"] += 1
            elif edge_x or edge_y:
                head_pos["靠边"] += 1
            else:
                head_pos["内部"] += 1
            # 撞的是身体哪段（死亡 head 落在动作前 body 的位置）
            dx, dy = DIRECTIONS[a]
            hit = (hx+dx, hy+dy)
            if hit in body_set:
                idx = list(e.body).index(hit)
                L = len(e.body)
                if idx < L/3:
                    hit_part["靠近头(前1/3)"] += 1
                elif idx < 2*L/3:
                    hit_part["中段"] += 1
                else:
                    hit_part["靠近尾(后1/3)"] += 1
            # 动作相对颈部
            if neck is not None:
                if (dx, dy) == (-neck[0], -neck[1]):
                    turn_type["掉头"] += 1
                elif (dx, dy) == neck:
                    turn_type["直行"] += 1
                else:
                    turn_type["转弯"] += 1
            break

print(f"共 {n} 次死亡")
print("死前剩余安全动作数:", dict(safe_count))
print("蛇头位置:", head_pos)
print("撞身体部位:", hit_part)
print("动作类型:", turn_type)
