import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from latent_world_model import LatentWorldModel
from train_rl import load_checkpoint

N_DATA, BATCH, STEPS = 20000, 64, 3000

beh = DQNAgent(state_mode="local")
load_checkpoint(beh, "dqn_ckpt/final.npz")
beh.eps = 0.3
S, A, R, S2 = [], [], [], []
e = SnakeEnv()
st = e.reset()
while len(S) < N_DATA:
    a = beh.act(st, training=True)
    s2, r, done, _ = e.step(a)
    S.append(st); A.append(a); R.append(r); S2.append(s2)
    st = s2
    if done:
        st = e.reset()
S = np.array(S, np.float32); A = np.array(A)
R = np.array(R, np.float32); S2 = np.array(S2, np.float32)
cut = int(0.9 * N_DATA)

wm = LatentWorldModel(state_dim=40, n_actions=4)
for _ in range(STEPS):
    idx = np.random.randint(0, cut, BATCH)
    wm.update_world(S[idx], A[idx], R[idx], S2[idx])

# 单步：编码->动力学->解码，对比真实下一观测
vi = np.arange(cut, N_DATA)
z = wm.encode(S[vi])
z2 = wm.step(z, A[vi])
o_pred = wm.decode(z2)
r_pred = wm.reward(z2)
disc = list(range(8))
for g in range(8):
    b = 8 + g * 4
    disc += [b + 1, b + 2, b + 3]
acc = (o_pred[:, disc] > 0.5) == (S2[vi][:, disc] > 0.5)
print(f"单步: 观测MSE={((o_pred-S2[vi])**2).mean():.4f} "
      f"奖励MSE={((r_pred-R[vi])**2).mean():.4f} 离散准确率={100*acc.mean():.1f}%")

# 多步隐空间想象（解码后对比真实观测）
horizons = [1, 2, 5, 10, 20]
err = {h: [] for h in horizons}
for seed in range(6000, 6020):
    env = SnakeEnv()
    c = env.reset(seed)
    states, actions = [c], []
    for _ in range(25):
        a = beh.act(c, training=False)
        c, _, done, _ = env.step(a)
        states.append(c); actions.append(a)
        if done:
            break
    if len(actions) < 20:
        continue
    z0 = wm.encode(np.asarray([states[0]]))
    zt = wm.rollout(z0[0], actions)
    for h in horizons:
        op = wm.decode(np.asarray([zt[h]]))[0]
        err[h].append(((op - states[h]) ** 2).mean())
print("隐空间多步想象误差:",
      {h: round(float(np.mean(v)), 4) for h, v in err.items()})
print("(对照 W1 观测空间: {1:0.007, 5:0.034, 10:0.172, 20:0.288})")
