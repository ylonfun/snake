import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")

normal_pred, normal_real, food_pred, food_real = [], [], [], []
for seed in range(10):
    env = SnakeEnv(); st = env.reset(seed)
    while True:
        a = ag.act(st, False)
        # 用真实当前编码做 R 预测（分布内，不依赖自回归）
        z = ag.world.encode(np.asarray([st], np.float32))
        rp = float(ag.world.R.forward(z)[0, 0])
        st2, r, done, info = env.step(a)
        if r == env.r_food:
            food_pred.append(rp); food_real.append(r)
        else:
            normal_pred.append(rp); normal_real.append(r)
        if done:
            break
        st = st2
print(f"普通步 {len(normal_pred)} 个：真实奖励均值 {np.mean(normal_real):.3f}，R预测均值 {np.mean(normal_pred):.3f}")
print(f"吃苹果步 {len(food_pred)} 个：真实 {np.mean(food_real):.1f}，R预测均值 {np.mean(food_pred):.2f}")
