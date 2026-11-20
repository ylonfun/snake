import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

# critic 校准：预测 V vs 真实折扣回报
pred_v, real_ret = [], []
for seed in range(5000, 5020):
    env = SnakeEnv(); st = env.reset(seed)
    z = ag.world.encode(np.asarray([st]))
    pred_v.append(ag.critic.forward(z)[0, 0])
    ret, disc = 0.0, 1.0
    while True:
        a = ag.act(st, training=False)
        st, r, done, _ = env.step(a)
        ret += disc * r; disc *= 0.95
        if done: break
    real_ret.append(ret)
pred_v, real_ret = np.array(pred_v), np.array(real_ret)
print("critic 预测 V: 均值", round(pred_v.mean(), 2), "范围", round(pred_v.min(), 2), round(pred_v.max(), 2))
print("真实折扣回报: 均值", round(real_ret.mean(), 2), "范围", round(real_ret.min(), 2), round(real_ret.max(), 2))
print("相关性:", round(np.corrcoef(pred_v, real_ret)[0, 1], 3))
