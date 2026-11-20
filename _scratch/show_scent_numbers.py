import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=120)
from rl_env import SnakeEnv

env = SnakeEnv(); env.reset(1)
env.food = (5, 5)            # 把苹果放在内部中心
env._build_scent()
print("苹果在(5,5)时的气味场（边界=最外一圈）：")
print(env.scent)
print("苹果格气味:", env.scent[5, 5])
print("距离1格(5,6):", env.scent[6, 5], " 距离2格(5,7):", env.scent[7, 5])
print("角落(0,0):", env.scent[0, 0], " 边界中点(0,5):", env.scent[0, 5])
print("相邻格气味差(中心附近):", round(env.scent[5,5]-env.scent[6,5], 5))
print("相邻格气味差(边界附近):", round(env.scent[0,5]-env.scent[1,5], 6))
