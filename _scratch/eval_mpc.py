import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
try:
    ag.load("dreamer_ckpt/latest.npz")
except Exception:
    pass
w = ag.world
GAMMA, DEPTH = 0.95, 8


def rollout_value(z, depth):
    """从 z 起在想象中用 critic 贪心 rollout，返回折扣回报。"""
    G, disc, cur = 0.0, 1.0, z
    for _ in range(depth):
        zn = w.step(np.repeat(cur, 4, 0), np.arange(4))
        q = w.R.forward(zn) + GAMMA * w.C.forward(zn) * ag.critic.forward(zn)
        b = int(q.argmax())
        G += disc * w.R.forward(zn)[b, 0]
        disc *= GAMMA
        cur = zn[b:b + 1]
        if w.C.forward(zn)[b, 0] < 0.5:
            break
    else:
        G += disc * ag.critic.forward(cur)[0, 0]
    return G


def mpc_action(state):
    z = w.encode(np.asarray([state]))
    zn = w.step(np.repeat(z, 4, 0), np.arange(4))
    r1, c1 = w.R.forward(zn), w.C.forward(zn)
    best, best_q = 0, -1e9
    for a in range(4):
        if c1[a, 0] < 0.5:
            q = r1[a, 0]
        else:
            q = r1[a, 0] + GAMMA * c1[a, 0] * rollout_value(zn[a:a + 1], DEPTH)
        if q > best_q:
            best_q, best = q, a
    return best


scores, steps_list = [], []
for seed in range(5000, 5010):
    env = SnakeEnv(); st = env.reset(seed)
    while True:
        st, r, done, info = env.step(mpc_action(st))
        if done:
            scores.append(info["score"]); steps_list.append(info["steps"]); break
print("MPC 规划 10 局: 均分", round(np.mean(scores), 1), "峰值", max(scores),
      "平均步数", round(np.mean(steps_list)))
