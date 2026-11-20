import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_lookahead_agent import CNNLookaheadAgent

N_GAMES = 12


def run(depth):
    agent = CNNLookaheadAgent(lookahead=depth)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    scores, death_len = [], []
    for seed in range(5000, 5000 + N_GAMES):
        env = SnakeEnv(state_mode="grid")
        s = env.reset(seed)
        while True:
            a = agent.act(s, training=False, env=env)
            s, _, done, info = env.step(a)
            if done:
                scores.append(info["score"]); death_len.append(len(env.body)); break
    return np.mean(scores), np.max(scores), np.mean(death_len)


for d in (3, 6, 10):
    m, mx, dl = run(d)
    print(f"前瞻深度 {d:2d}  均分={m:.1f} 峰值={mx} 死亡体长={dl:.1f}")
