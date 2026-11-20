import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
from rl_env import SnakeEnv

# 期限设为3，蛇在空旷处安全直行、食物远在左上角吃不到
env = SnakeEnv(food_deadline=3)
env.reset(1)
env.body = [(5, 5)]; env.direction = (1, 0)
env.food = (0, 0)
for t in range(6):
    _, r, done, info = env.step(3)   # 一直向右，安全不撞
    print(f"第{t+1}步 饥饿计数={env.steps_since_food} 结束={done} 得分={info['score']}")
    if done:
        break
