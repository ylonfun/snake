"""最终对比（30 局）：单步贪心 / W4 前瞻 / W6 安全空间。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_lookahead_agent import CNNLookaheadAgent
from cnn_safe_agent import CNNSafeAgent

w4 = CNNLookaheadAgent(lookahead=10)
w4.load(os.path.join("cnn_ckpt", "best.npz"))
w6 = CNNSafeAgent()
w6.load(os.path.join("cnn_ckpt", "best.npz"))


def run(mode):
    scores, death_len, steps = [], [], []
    for seed in range(5000, 5030):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        n = 0
        while True:
            if mode == "single":
                a = w6.act(s, training=False)
            elif mode == "w4":
                a = w4.act(s, training=False, env=env)
            else:
                a = w6.act(s, training=False, env=env)
            n += 1
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body))
                steps.append(n); break
    return np.mean(scores), np.max(scores), np.mean(death_len), np.mean(steps)


for name, mode in [("单步贪心", "single"), ("W4 前瞻", "w4"),
                   ("W6 安全空间", "w6")]:
    m, mx, dl, st = run(mode)
    print(f"{name:10s} 均分={m:.1f} 峰值={mx} 死亡体长={dl:.1f} 平均步数={st:.0f}")
