import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")
wm = ag.world

env = SnakeEnv(); st = env.reset(5000)
# 手动复现到沿右边界向下的状态
env.body = [(9, 3)]; env.direction = (0, 1)
for k in range(4):
    s = env.get_state()
    z = wm.encode(np.asarray([s]))
    z_next = wm.step(z, np.array([1]))          # 想象"向下"
    p_cont = wm.cont_prob(z_next)               # 想象后的继续概率
    r_pred = wm.reward(z_next)
    print(f"头{env.body[0]} 想象向下→ 继续概率={float(p_cont[0]):.3f} 预测奖励={float(r_pred[0]):.3f}")
    s2, _, done, _ = env.step(1)
    if done:
        print("真实结果：撞墙死亡")
        break
