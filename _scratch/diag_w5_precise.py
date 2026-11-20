"""精确诊断：z 可分性 + 重建质量 + 奖励头单独深训。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from mlp import MLP
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
model = LatentGridModel(); model.load(os.path.join("cnn_ckpt", "world_cnn.npz"))
env = SnakeEnv(state_mode="grid")
G, G2, R, D = [], [], [], []
s = env.reset()
while len(G) < 3000:
    a = behavior.act(s, training=False)
    s2, r, done, _ = env.step(a)
    G.append(s); G2.append(s2); R.append(r); D.append(done)
    s = env.reset() if done else s2
G = np.asarray(G, np.float32); G2 = np.asarray(G2, np.float32)
R = np.asarray(R, np.float32); D = np.asarray(D)
z = model.encode(G2)
food = R > 5

# (1) z 可分性
lp = MLP([64, 1], lr=1e-2); y = food.astype(np.float32)
for _ in range(800):
    lp.zero_grad(); g = (lp.forward(z) - y.reshape(-1, 1)) / len(z)
    lp.backward(g); lp.adam_step()
print("(1) z 对吃苹果线性探针相关性:",
      round(np.corrcoef(lp.forward(z)[:, 0], y)[0, 1], 3))

# (2) 重建质量（逐通道）
recon = model.reconstruct(model.encode(G))
names = ["头", "身体", "食物", "气味", "访问", "边界"]
for c in range(6):
    print(f"(2) 重建 {names[c]} 通道 MSE={np.mean((recon[:, c]-G[:, c])**2):.4f}")

# (3) 奖励头在固定 z 上单独深训（全批量，3000 步）
pr = MLP([64, 64, 1], lr=1e-3)
for _ in range(3000):
    pr.zero_grad(); g = (pr.forward(z) - R.reshape(-1, 1)) / len(z)
    pr.backward(g); pr.adam_step()
p = pr.forward(z)[:, 0]
print("(3) 单独深训奖励头: 吃苹果预测", round(p[food].mean(), 2),
      "普通", round(p[~food].mean(), 3), "整体MSE", round(np.mean((p-R)**2), 3))
