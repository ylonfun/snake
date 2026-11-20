"""测量在线训练时各头在 mini-batch 下的梯度 norm 与裁剪频率。"""
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
G2 = np.asarray(G2, np.float32); R = np.asarray(R, np.float32); D = np.asarray(D)
N = len(G2)


def norm_of(m):
    t = sum(np.sum(g ** 2) for g in m.gW) if isinstance(m.gW, list) \
        else np.sum(m.gW ** 2)
    t += sum(np.sum(g ** 2) for g in m.gb) if isinstance(m.gb, list) \
        else np.sum(m.gb ** 2)
    return np.sqrt(t)


stats = {"rew": [], "cont": [], "dyn": []}
for _ in range(200):
    idx = np.random.randint(0, N, 64)
    z2 = model.encode(G2[idx])
    # 奖励头
    model.rew.zero_grad()
    g = (model.rew.forward(z2) - R[idx].reshape(-1, 1)) / 64
    model.rew.backward(g); stats["rew"].append(norm_of(model.rew))
    # 继续头
    model.cont.zero_grad()
    pc = model.cont.forward(z2)
    tc = (1 - D[idx]).reshape(-1, 1)
    w = np.where(tc > .5, 1.0, 30.0)
    model.cont.backward((pc - tc) * w / 64); stats["cont"].append(norm_of(model.cont))

for k, v in stats.items():
    v = np.array(v)
    print(f"{k}: norm 中位={np.median(v):.2f} 最大={v.max():.2f} "
          f"超过5的比例={np.mean(v > 5)*100:.0f}%")
