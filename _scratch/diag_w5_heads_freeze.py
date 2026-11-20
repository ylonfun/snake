"""诊断头部欠拟合：固定数据集上冻结编码器，只训练 dyn/rew/cont，看头能否学好。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
env = SnakeEnv(state_mode="grid")
G, A, R, G2, D = [], [], [], [], []
s = env.reset()
while len(G) < 6000:
    a = behavior.act(s, training=False)
    s2, r, done, _ = env.step(a)
    G.append(s); A.append(a); R.append(r); G2.append(s2); D.append(done)
    s = env.reset() if done else s2
G, G2 = np.asarray(G, np.float32), np.asarray(G2, np.float32)
A, R, D = np.asarray(A), np.asarray(R, np.float32), np.asarray(D)
N = len(G)
split = 5000  # 前 5000 训练，后 1000 留出

model = LatentGridModel(z_dim=64)
model.train_encoder = False
for step in range(1, 4001):
    idx = np.random.randint(0, split, 64)
    model.update(G[idx], A[idx], R[idx], G2[idx], D[idx])

# 留出评估
te = np.arange(split, N)
z2 = model.encode(G2[te])
pr = model.reward_of(z2); pc = model.cont_prob(z2)
rt, dt = R[te], D[te]
food, death = rt > 5, rt < -5
normal = ~food & ~death
print("冻结编码器后，奖励头（预测/真实）:")
print("  吃苹果:", round(pr[food].mean(), 2), "/", round(rt[food].mean(), 2),
      "n=", food.sum())
print("  死亡  :", round(pr[death].mean(), 2), "/", round(rt[death].mean(), 2),
      "n=", death.sum())
print("  普通步:", round(pr[normal].mean(), 3), "/", round(rt[normal].mean(), 3))
print("继续头 死亡样本存活概率:", round(pc[dt].mean(), 3),
      "存活样本:", round(pc[~dt].mean(), 3))
