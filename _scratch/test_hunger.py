import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv

# 测试1：随机游走，记录死亡原因与步数
env = SnakeEnv()
st = env.reset(1)
for t in range(200):
    a = np.random.randint(4)
    st, r, done, info = env.step(a)
    if done:
        print(f"随机游走 第{t+1}步结束 得分={info['score']} 饥饿计数={env.steps_since_food}")
        break

# 测试2：手动连续吃苹果，验证计数清零、不会饿死
env2 = SnakeEnv()
env2.reset(1)
env2.body = [(2, 2)]; env2.direction = (1, 0)
env2.food = (3, 2)
env2.steps_since_food = 55            # 模拟已接近期限
_, r, done, info = env2.step(3)       # 向右吃到苹果
print("吃苹果后 饥饿计数:", env2.steps_since_food, "奖励:", r, "是否结束:", done)

# 测试3：关闭饥饿（food_deadline=None）时不因此死亡
env3 = SnakeEnv(food_deadline=None)
env3.reset(1)
print("food_deadline =", env3.food_deadline)
