import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(learn_start=50)
env = SnakeEnv()
st = env.reset(seed=1)
for t in range(500):
    a = ag.act(st, training=True)
    s2, r, done, _ = env.step(a)
    ag.remember(st, a, r, s2, done)
    ag.learn()
    st = s2
    if done:
        st = env.reset()
print("冒烟通过 step=", ag.step_count)
# 检查 actor 未饱和、critic 有界
z = ag.world.encode(np.asarray([env.reset(seed=2)]))
print("actor logits=", np.round(ag.actor.forward(z)[0], 2),
      "critic V=", round(ag.critic.forward(z)[0, 0], 2))
