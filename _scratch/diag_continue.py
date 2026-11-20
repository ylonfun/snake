import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent()
ag.load("dreamer_ckpt/latest.npz")

for seed in range(5000, 5003):
    env = SnakeEnv()
    st = env.reset(seed)
    # 找致命动作
    fatal_a = None
    for a in range(4):
        e2 = SnakeEnv(); e2.reset(seed)
        _, r_real, done, _ = e2.step(a)
        if done:
            fatal_a = a; break
    z = ag.world.encode(np.asarray([st]))
    z2 = ag.world.step(z, np.asarray([fatal_a]))
    r_pred = ag.world.R.forward(z2)[0, 0]
    print(f"seed{seed} 致命动作={fatal_a} 真实r={r_real} 模型想象r_pred={r_pred:.2f} "
          f"模型是否知道终止: 无终止预测头，想象继续")
