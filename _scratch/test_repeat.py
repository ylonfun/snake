import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
from rl_env import SnakeEnv

env = SnakeEnv()
env.reset(1)
# 手动构造 2×2 区域：头在 (2,2)，方向向右
env.body = [(2, 2)]
env.direction = (1, 0)
env.visit_count[:] = 0
env.visit_count[2, 2] = 1
# 动作序列：右、下、左、上 重复（每步90度转弯，不反向）
seq = [3, 1, 2, 0]
for lap in range(4):
    rewards = []
    for a in seq:
        _, r, done, _ = env.step(a)
        rewards.append(round(r, 3))
    print(f"第{lap+1}圈 每步奖励:", rewards)
