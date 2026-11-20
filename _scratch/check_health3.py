"""训练代码路径微冒烟：只跑内存内 learn，不触碰任何存档文件。"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np

from rl_env import SnakeEnv

# ---- DQN 40 维 learn ----
from dqn_agent import DQNAgent
agent = DQNAgent(learn_start=20, batch_size=16)
env = SnakeEnv()
s = env.reset(1)
losses = []
for _ in range(120):
    a = agent.act(s, training=True)
    s2, r, done, _ = env.step(a)
    agent.remember(s, a, r, s2, done)
    lo = agent.learn()
    if lo is not None:
        losses.append(lo)
    s = env.reset() if done else s2
assert len(losses) > 50 and np.isfinite(losses[-1])
print(f"[PASS] DQN learn 120 步：{len(losses)} 次更新 last_loss={losses[-1]:.4f}")

# ---- CNN-DQN learn（从零，小步数）----
from cnn_dqn_agent import CNNDQNAgent
cagent = CNNDQNAgent(learn_start=20, batch_size=16)
env = SnakeEnv(state_mode="grid")
g = env.reset(2)
cl = []
for _ in range(60):
    a = cagent.act(g, training=True)
    g2, r, done, _ = env.step(a)
    cagent.remember(g, a, r, g2, done)
    lo = cagent.learn()
    if lo is not None:
        cl.append(lo)
    g = env.reset() if done else g2
assert len(cl) > 20
print(f"[PASS] CNN-DQN learn 60 步：{len(cl)} 次更新")

# ---- Dreamer：世界模型更新 + 想象 ----
from dreamer_agent import DreamerAgent
d = DreamerAgent(state_dim=42, imag_batch=4, batch_size=16, learn_start=20, horizon=3)
env = SnakeEnv()
s = env.reset(3)
for _ in range(80):
    a = d.act(s, training=True)
    s2, r, done, _ = env.step(a)
    d.remember(s, a, r, s2, done)
    d.learn()
    s = env.reset() if done else s2
print(f"[PASS] Dreamer 世界模型更新+想象 80 步 步数计数={d.step_count}")

# ---- GridForwardModel 前向世界模型一步更新 ----
from grid_forward_model import GridForwardModel
m = GridForwardModel()
env = SnakeEnv(state_mode="grid")
G, A, G2, D = [], [], [], []
g = env.reset(4)
for _ in range(16):
    a = int(np.random.randint(4))
    g2, r, done, _ = env.step(a)
    G.append(g); A.append(a); G2.append(g2); D.append(done)
    g = env.reset() if done else g2
stat = m.update(np.asarray(G, np.float32), np.asarray(A),
                np.asarray(G2, np.float32), np.asarray(D, np.float32))
print(f"[PASS] grid 前向模型 update：{stat}")

# ---- LatentGridModel 一步更新 ----
from latent_grid_model import LatentGridModel
lm = LatentGridModel(z_dim=16)
G, A, R, G2, D = [np.asarray(x, np.float32) for x in (G,)], None, None, None, None
g = env.reset(5)
sg, sa, sr, sg2, sd = [], [], [], [], []
for _ in range(8):
    a = int(np.random.randint(4))
    g2, r, done, _ = env.step(a)
    sg.append(g); sa.append(a); sr.append(r); sg2.append(g2); sd.append(done)
    g = env.reset() if done else g2
stat2 = lm.update(np.asarray(sg, np.float32), np.asarray(sa),
                  np.asarray(sr, np.float32), np.asarray(sg2, np.float32),
                  np.asarray(sd, np.float32))
print(f"[PASS] 潜在 grid 世界模型 update：{stat2}")

print("OK")
