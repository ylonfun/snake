import os
from collections import Counter

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(learn_start=50)
env = SnakeEnv()
st = env.reset(1)
aa = []
for t in range(800):
    a = ag.act(st, True)
    aa.append(a)
    s2, r, done, _ = env.step(a)
    ag.remember(st, a, r, s2, done)
    ag.learn()
    st = s2
    if done:
        st = env.reset()
print("冒烟通过，动作分布:", Counter(aa))
z = ag.world.encode(np.asarray([env.reset(2)]))
print("logits:", np.round(ag.actor.forward(z)[0], 2))
