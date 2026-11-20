import os
from collections import Counter

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent()
ag.load("dreamer_ckpt/latest.npz")
env = SnakeEnv()
st = env.reset(5000)
acts, dists, toward = [], [], []
for t in range(200):
    hx, hy = env.body[0]
    fx, fy = env.food
    dists.append(abs(hx-fx) + abs(hy-fy))
    a = ag.act(st, training=False)
    acts.append(a)
    # 该动作是否缩短到食物距离
    from settings import DIRECTIONS
    d = DIRECTIONS[a]
    nd = abs(hx+d[0]-fx) + abs(hy+d[1]-fy)
    toward.append(nd < dists[-1])
    st, _, done, _ = env.step(a)
    if done:
        print("done at", t); break
print("动作分布:", Counter(acts))
print("距离前20:", dists[:20])
print("朝食物移动比例:", np.mean(toward))
# 看一个见食物状态的 logits
env2 = SnakeEnv(); s = env2.reset(5000)
z = ag.world.encode(np.asarray([s]))
print("actor logits:", np.round(ag.actor.forward(z)[0], 2))
print("critic V:", round(ag.critic.forward(z)[0, 0], 2))
