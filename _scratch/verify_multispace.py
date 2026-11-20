"""多步空间规划 30 局权重扫描。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from _scratch.exp_multistep_space import MultiSpaceAgent


def run(weight):
    agent = MultiSpaceAgent(space_weight=weight)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, death_len, steps = [], [], []
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
                steps.append(n); break
    return np.mean(scores), np.max(scores), np.mean(death_len), np.mean(steps)


for w in (4.0, 6.0, 8.0, 10.0):
    m, mx, dl, st = run(w)
    print(f"多步空间 权重{w:4.1f} 均分={m:.1f} 峰值={mx} 死亡体长={dl:.1f} 步数={st:.0f}")
