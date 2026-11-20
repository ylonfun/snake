"""验证保分辨率前向模型：单步泛化 + 多步自回归。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from grid_forward_model import GridForwardModel
from cnn_dqn_agent import CNNDQNAgent

behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
env = SnakeEnv(state_mode="grid")
G, A, G2, D = [], [], [], []
s = env.reset(3000)
while len(G) < 6000:
    a = behavior.act(s, training=False)
    s2, _, done, _ = env.step(a)
    G.append(s); A.append(a); G2.append(s2); D.append(done)
    s = env.reset(3000 + len(G)) if done else s2
G = np.asarray(G, np.float32); G2 = np.asarray(G2, np.float32)
A = np.asarray(A); D = np.asarray(D)
split = 4500
tr, va = np.arange(split), np.arange(split, 6000)

model = GridForwardModel(lr=1e-3)
for step in range(1, 8001):
    idx = tr[np.random.randint(0, split, 64)]
    model.update(G[idx], A[idx], G2[idx], D[idx])
    if step % 2000 == 0:
        pvg, pvd = model.forward(G[va], A[va])
        mse = np.mean((pvg - G2[va]) ** 2)
        acc = np.mean((pvd > .5) == D[va])
        print(f"步{step} 验证grid MSE={mse:.4f} done准确率={acc:.3f}")


def multistep(n_ep=8, horizon=10):
    errs = {h: [] for h in range(1, horizon + 1)}
    for ep in range(n_ep):
        e = SnakeEnv(state_mode="grid"); g = e.reset(8000 + ep)
        real, acts = [g], []
        for _ in range(horizon):
            a = behavior.act(g, training=False)
            g, _, done, _ = e.step(a)
            acts.append(a); real.append(g)
            if done: break
        cur = real[0]
        for t in range(len(acts)):
            out, _ = model.forward(cur[np.newaxis], [acts[t]])
            cur = out[0]
            errs[t + 1].append(np.mean((cur - real[t + 1]) ** 2))
    return {h: np.mean(v) for h, v in errs.items() if v}


model.save(os.path.join("cnn_ckpt", "grid_forward.npz"))
print("多步自回归误差:", {h: round(v, 4) for h, v in multistep().items()})
