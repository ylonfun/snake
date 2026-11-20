import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv

env = SnakeEnv()
env.reset(1)
env.body = [(5, 5)]; env.food = (5, 6); env.direction = (0, 1)
env._build_scent()
print("状态维度:", env.get_state().shape[0])
print("苹果格(5,6)气味:", round(float(env.scent[6, 5]), 3),
      " 距离1格(5,7):", round(float(env.scent[7, 5]), 3),
      " 距离2格(5,8):", round(float(env.scent[8, 5]), 3),
      " 距离3格(5,9):", round(float(env.scent[9, 5]), 3))

# 连续两次进入同一个有气味的格子，验证气味奖励递减
env.body = [(5, 5)]; env.direction = (0, 1); env.food = (5, 8)
env._build_scent()
for k in range(3):
    env.body = [(5, 5)]; env.direction = (0, 1)
    _, r, _, _ = env.step(1)   # 走到 (5,6)
    print(f"第{k+1}次进入(5,6) 该步奖励={round(float(r),4)} 该格剩余气味={round(float(env.scent[6,5]),4)}")
