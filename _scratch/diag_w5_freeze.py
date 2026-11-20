"""W5 稳定性诊断：固定数据集上对比 冻结编码器 vs 全更新。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

# 收集固定数据集
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


def multi_step_err(model, n_ep=6, horizon=5):
    errs = {h: [] for h in range(1, horizon + 1)}
    for ep in range(n_ep):
        e = SnakeEnv(state_mode="grid"); g = e.reset(7000 + ep)
        tg, ta = [g], []
        for _ in range(horizon):
            a = behavior.act(g, training=False)
            g, _, done, _ = e.step(a)
            ta.append(a); tg.append(g)
            if done: break
        rz = [model.encode(np.asarray([x], np.float32)) for x in tg]
        z = rz[0]
        for t in range(len(ta)):
            z = model.step_latent(z, [ta[t]])
            errs[t + 1].append(np.mean((z - rz[t + 1]) ** 2))
    return {h: np.mean(v) for h, v in errs.items() if v}


def run(freeze):
    model = LatentGridModel(z_dim=64)
    model.train_encoder = not freeze
    for step in range(1, 4001):
        idx = np.random.randint(0, N, 64)
        model.update(G[idx], A[idx], R[idx], G2[idx], D[idx])
        if step % 1000 == 0:
            me = multi_step_err(model)
            tag = "冻结编码器" if freeze else "全更新"
            print(f"[{tag}] 步{step} " + " ".join(f"h{h}={v:.4f}"
                  for h, v in me.items()))


print("="  * 20, "模式一：冻结编码器")
run(True)
print("=" * 20, "模式二：全更新（对照）")
run(False)
