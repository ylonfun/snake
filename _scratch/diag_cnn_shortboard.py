import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from settings import DIRECTIONS
from rl_env import SnakeEnv
from cnn_dqn_agent import CNNDQNAgent

agent = CNNDQNAgent()
agent.load(os.path.join("cnn_ckpt", "final.npz"))

def safe_actions(env):
    hx, hy = env.body[0]
    safe = []
    for a in range(4):
        d = DIRECTIONS[a]
        nh = (hx + d[0], hy + d[1])
        if nh[0] < 0 or nh[0] >= env.cols or nh[1] < 0 or nh[1] >= env.rows:
            continue
        if nh in env.body[:-1]:
            continue
        safe.append(a)
    return safe

last_safe, causes, dlen, scores, seg_ratios = [], [], [], [], []
for seed in range(5000, 5030):
    env = SnakeEnv(state_mode="grid")
    s = env.reset(seed)
    seg_origin = env.body[0]
    seg_steps = 0
    while True:
        n_safe = len(safe_actions(env))
        target_food = env.food                 # 本步瞄准的（旧）苹果
        a = agent.act(s, training=False)
        s, r, done, info = env.step(a)
        seg_steps += 1
        if r > 5:  # 吃到 target_food
            man = (abs(target_food[0]-seg_origin[0])
                   + abs(target_food[1]-seg_origin[1]))
            seg_ratios.append(seg_steps / max(1, man))
            seg_origin = env.body[0]; seg_steps = 0
        if done:
            last_safe.append(n_safe); causes.append(env.death_cause)
            dlen.append(len(env.body)); scores.append(info["score"]); break

last_safe, dlen, scores = np.array(last_safe), np.array(dlen), np.array(scores)
print("均分:", round(scores.mean(), 1), "死亡体长均值:", round(dlen.mean(), 1))
from collections import Counter
print("死因:", dict(Counter(causes)))
print("死前安全动作数分布:", dict(Counter(last_safe.tolist())))
print("  真困死(0个)占比:", round((last_safe == 0).mean()*100), "%")
print("  判断失误(>0个)占比:", round((last_safe > 0).mean()*100), "%")
print("路径比(实际/曼哈顿) 均值:", round(np.mean(seg_ratios), 2),
      "中位:", round(np.median(seg_ratios), 2), "最差:", round(np.max(seg_ratios), 2))
