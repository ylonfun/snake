"""W6 最终定位：30 局大样本，权重 10/12/14。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_safe_agent import CNNSafeAgent


def run(weight):
    agent = CNNSafeAgent(lookahead=10, space_weight=weight)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, death_len, steps, causes = [], [], [], []
    for seed in range(5000, 5030):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        n = 0
        while True:
            a = agent.act(s, training=False, env=env)
            n += 1
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body))
                steps.append(n); causes.append(env.death_cause); break
    return (np.mean(scores), np.max(scores), np.mean(death_len),
            np.mean(steps), {c: causes.count(c) for c in set(causes)})


for w in (10.0, 12.0, 14.0):
    m, mx, dl, st, causes = run(w)
    print(f"权重{w:4.1f} 均分={m:.1f} 峰值={mx} 死亡体长={dl:.1f} "
          f"平均步数={st:.0f} 死因={causes}")
