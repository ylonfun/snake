import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from world_model import WorldModel
from train_rl import load_checkpoint

N_DATA = 20000
BATCH = 64
STEPS = 3000

# 行为策略：已训 DQN + 0.3 随机，覆盖长短身体
beh = DQNAgent(state_mode="local")
load_checkpoint(beh, "dqn_ckpt/final.npz")
beh.eps = 0.3

S, A, R, S2, D = [], [], [], [], []
e = SnakeEnv()
st = e.reset()
while len(S) < N_DATA:
    a = beh.act(st, training=True)
    s2, r, done, info = e.step(a)
    S.append(st); A.append(a); R.append(r); S2.append(s2); D.append(done)
    st = s2
    if done:
        st = e.reset()

S = np.array(S, np.float32); A = np.array(A); R = np.array(R, np.float32)
S2 = np.array(S2, np.float32); D = np.array(D)
cut = int(0.9 * N_DATA)

wm = WorldModel(state_dim=40, n_actions=4)
for t in range(STEPS):
    idx = np.random.randint(0, cut, BATCH)
    wm.train_step(S[idx], A[idx], R[idx], S2[idx])

# --- 单步预测（验证集）---
vi = np.arange(cut, N_DATA)
x = wm._inputs(S[vi], A[vi])
pred_s = wm.dynamics.forward(x)
pred_r = wm.reward.forward(x)[:, 0]
mse_state = ((pred_s - S2[vi]) ** 2).mean()
mse_rew = ((pred_r - R[vi]) ** 2).mean()
# 离散位（前8维方向 + 各 see_*/adj_self）用 0.5 阈值
disc = list(range(8))
for g in range(8):
    b = 8 + g * 4
    disc += [b + 1, b + 2, b + 3]
acc = (pred_s[:, disc] > 0.5) == (S2[vi][:, disc] > 0.5)
print(f"单步(验证集{len(vi)}条): 状态MSE={mse_state:.4f} 奖励MSE={mse_rew:.4f} "
      f"离散特征准确率={100*acc.mean():.1f}%")

# --- 多步 rollout：真实动作序列，模型纯想象 ---
horizons = [1, 2, 5, 10, 20]
err = {h: [] for h in horizons}
for seed in range(6000, 6020):
    env = SnakeEnv()
    s0 = env.reset(seed)
    states, actions = [s0], []
    cur = s0
    for _ in range(25):
        a = beh.act(cur, training=False)
        cur, _, done, _ = env.step(a)
        states.append(cur); actions.append(a)
        if done:
            break
    if len(actions) < 20:
        continue
    pred = wm.rollout(states[0], actions)
    for h in horizons:
        err[h].append(((pred[h] - states[h]) ** 2).mean())

print("多步想象误差(随horizon累积):",
      {h: round(float(np.mean(v)), 4) for h, v in err.items()})
