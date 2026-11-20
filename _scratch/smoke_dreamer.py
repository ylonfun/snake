import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(learn_start=50)
env = SnakeEnv()
st = env.reset(seed=1)
for t in range(400):
    a = ag.act(st, training=True)
    s2, r, done, _ = env.step(a)
    ag.remember(st, a, r, s2, done)
    ag.learn()
    st = s2
    if done:
        st = env.reset()
print("冒烟通过，step_count=", ag.step_count)
print("贪心动作:", ag.act(env.reset(seed=2), training=False))
ag.save("dreamer_smoke.npz")
ag.load("dreamer_smoke.npz")
print("save/load 通过，step=", ag.step_count)
os.remove("dreamer_smoke.npz")
