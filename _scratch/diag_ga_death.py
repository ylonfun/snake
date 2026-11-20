import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from nn import Net
from settings import N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT

genes = np.array(list(map(float, open(os.path.join("genes", "record", "genes")).read().split())))
net = Net(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT, genes)

from collections import Counter
causes = Counter(); rows = []
for seed in range(30):
    env = SnakeEnv(); st = env.reset(seed)
    while True:
        a = net.predict(st[:32])      # GA 用原始 32 维
        st, r, done, info = env.step(a)
        if done:
            causes[env.death_cause] += 1
            rows.append((info["score"], env.death_cause))
            break
print("死因分布：", dict(causes))
print("各局(分数,死因)：", sorted(rows, reverse=True))
