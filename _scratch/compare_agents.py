"""基于 SnakeEnv 的三方法统一对比（10×10，5 局，相同 seed）。"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_dqn_agent import CNNDQNAgent
from cnn_safe_agent import CNNMultiSpaceAgent
from cnn_beam_agent import CNNBeamAgent


def evaluate(act_fn, n=5):
    scores, nsteps_list, causes, t_act = [], [], [], []
    for seed in range(5000, 5000 + n):
        env = SnakeEnv(state_mode="grid", deadline_growth=1.0)
        s = env.reset(seed); nsteps = 0
        while True:
            t0 = time.time()
            a = act_fn(s, env)
            t_act.append((time.time()-t0)*1000)
            nsteps += 1
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"])
                nsteps_list.append(nsteps)
                causes.append(env.death_cause)
                break
    uniq, counts = np.unique(causes, return_counts=True)
    return (np.mean(scores), np.max(scores), np.mean(nsteps_list),
            np.mean(t_act), dict(zip(uniq, counts)))


dqn = CNNDQNAgent(); dqn.load(os.path.join("cnn_ckpt", "best.npz"))
multi = CNNMultiSpaceAgent(); multi.load(os.path.join("cnn_ckpt", "best.npz"))
beam = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
beam.load(os.path.join("cnn_ckpt", "best.npz"))

configs = {
    "单步 CNN-DQN": (dqn, lambda s, env: dqn.act(s, training=False)),
    "实时多步空间": (multi, lambda s, env: multi.act(s, training=False, env=env)),
    "离线 beam": (beam, lambda s, env: beam.act(s, training=False, env=env)),
}
for name, (_, fn) in configs.items():
    m, mx, steps, ms, causes = evaluate(fn)
    print(f"{name}: 均分{m:.1f} 峰值{mx} 平均步数{steps:.0f} "
          f"每决策{ms:.1f}ms 死因{causes}")
