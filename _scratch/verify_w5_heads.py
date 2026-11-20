"""W5 头语义验证：奖励头能否区分吃苹果/死亡/普通步，继续头能否识别死亡。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
model = LatentGridModel(); model.load(os.path.join("cnn_ckpt", "world_cnn.npz"))

env = SnakeEnv(state_mode="grid")
G2, R, D = [], [], []
s = env.reset()
while len(G2) < 3000:
    a = behavior.act(s, training=False)
    s2, r, done, _ = env.step(a)
    G2.append(s2); R.append(r); D.append(done)
    s = env.reset() if done else s2
G2 = np.asarray(G2, np.float32); R = np.asarray(R); D = np.asarray(D)

z2 = model.encode(G2)
pred_r = model.reward_of(z2)
pred_c = model.cont_prob(z2)

food, death = R > 5, R < -5
normal = ~food & ~death
print("奖励头预测（预测均值 / 真实均值）:")
print("  吃苹果:", round(pred_r[food].mean(), 2), "/", round(R[food].mean(), 2),
      "n=", food.sum())
print("  死亡  :", round(pred_r[death].mean(), 2), "/", round(R[death].mean(), 2),
      "n=", death.sum())
print("  普通步:", round(pred_r[normal].mean(), 3), "/", round(R[normal].mean(), 3),
      "n=", normal.sum())

print("继续头（存活概率）:")
print("  真实死亡样本预测均值:", round(pred_c[D].mean(), 3), "n=", D.sum())
print("  真实存活样本预测均值:", round(pred_c[~D].mean(), 3))
pred_death = pred_c < 0.5
tp = (pred_death & D).sum(); fp = (pred_death & ~D).sum(); fn = (~pred_death & D).sum()
print(f"  死亡识别 命中={tp} 误报={fp} 漏报={fn}")
