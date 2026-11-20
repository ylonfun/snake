"""阶段一：独立训练并冻结隐空间世界模型，验证想象质量后保存。"""
import os
import time

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from latent_world_model import LatentWorldModel
from train_rl import load_checkpoint

N_DATA, BATCH, STEPS = 50000, 64, 12000

beh = DQNAgent(state_mode="local")
load_checkpoint(beh, "dqn_ckpt/final.npz")
beh.eps = 0.3
S, A, R, S2, D = [], [], [], [], []
e = SnakeEnv()
st = e.reset()
while len(S) < N_DATA:
    a = beh.act(st[:40], training=True)
    s2, r, done, _ = e.step(a)
    S.append(st); A.append(a); R.append(r); S2.append(s2); D.append(done)
    st = s2
    if done:
        st = e.reset()
S = np.array(S, np.float32); A = np.array(A); R = np.array(R, np.float32)
S2 = np.array(S2, np.float32); D = np.array(D, np.float32)
cut = int(0.9 * N_DATA)

wm = LatentWorldModel(state_dim=42, n_actions=4)
wall = time.time()
for t in range(STEPS):
    idx = np.random.randint(0, cut, BATCH)
    wm.update_world(S[idx], A[idx], R[idx], S2[idx], D[idx])
    if (t + 1) % 1000 == 0:
        print(f"  世界模型训练 {t+1}/{STEPS} 耗时={time.time()-wall:.0f}s")

# 验证多步想象（解码后对比真实观测）
vi = np.arange(cut, N_DATA)
o_pred = wm.decode(wm.step(wm.encode(S[vi]), A[vi]))
horizons = [1, 5, 10, 20]
err = {h: [] for h in horizons}
for seed in range(6000, 6020):
    env = SnakeEnv(); c = env.reset(seed)
    states, actions = [c], []
    for _ in range(25):
        a = beh.act(c[:40], False)
        c, _, done, _ = env.step(a)
        states.append(c); actions.append(a)
        if done: break
    if len(actions) < 20: continue
    zt = wm.rollout(wm.encode(np.asarray([states[0]]))[0], actions)
    for h in horizons:
        op = wm.decode(np.asarray([zt[h]]))[0]
        err[h].append(((op-states[h])**2).mean())
print("单步观测MSE:", round(((o_pred-S2[vi])**2).mean(), 4))
print("多步想象误差:", {h: round(float(np.mean(v)), 4) for h, v in err.items()})

# 保存世界模型
np.savez("world_frozen.npz",
         **{f"{tag}_W{i}": w for tag, net in [("E", wm.E), ("D", wm.D), ("R", wm.R),
                                              ("C", wm.C), ("Dec", wm.Dec), ("Et", wm.E_t)]
            for i, w in enumerate(net.W)},
         **{f"{tag}_b{i}": x for tag, net in [("E", wm.E), ("D", wm.D), ("R", wm.R),
                                              ("C", wm.C), ("Dec", wm.Dec), ("Et", wm.E_t)]
            for i, x in enumerate(net.b)})
print("世界模型已保存到 world_frozen.npz")
