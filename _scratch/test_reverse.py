import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
from rl_env import SnakeEnv

env = SnakeEnv()
env.reset(1)
# 强制方向向右，头在 (0,0)
env.direction = (1, 0)
env.body = [(0, 0)]
# 给反向动作 2=左，应被忽略、保持向右直行
s, r, done, info = env.step(2)
print("反向输入后实际头位置:", env.body[0], "(应仍向右到 (1,0))", "方向:", env.direction)
# 连续给反向，确认不会回到 (0,0)
env.step(2)
print("再次反向后头部:", env.body[0], "(应到 (2,0))，未发生两格往返")
