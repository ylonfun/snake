"""全批量补训奖励头：固定 z，lr=1e-3，每步一个完整 epoch，快速拟合。"""
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
while len(G2) < 4000:
    a = behavior.act(s, training=False)
    s2, r, done, _ = env.step(a)
    G2.append(s2); R.append(r); D.append(done)
    s = env.reset() if done else s2
G2 = np.asarray(G2, np.float32); R = np.asarray(R, np.float32); D = np.asarray(D)
z = model.encode(G2)
food, death, normal = R > 5, R < -5, ~(R > 5) & ~(R < -5)

model.rew.lr = 1e-3
for step in range(1, 3001):
    model.rew.zero_grad()
    g = (model.rew.forward(z) - R.reshape(-1, 1)) / len(R)
    model.rew.backward(g); model.rew.adam_step()
    if step % 500 == 0:
        p = model.reward_of(z)
        print("步%d 吃苹果=%.2f 死亡=%.2f 普通=%.3f" % (
            step, p[food].mean(), p[death].mean(), p[normal].mean()))

model.save(os.path.join("cnn_ckpt", "world_cnn.npz"))
print("全批量补训完成并保存")
