import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True, learn_start=50)
ag.load_world("world_frozen.npz")
env = SnakeEnv(); st = env.reset(1)
for t in range(600):
    a = ag.act(st, True)
    s2, r, done, _ = env.step(a)
    ag.remember(st, a, r, s2, done); ag.learn(); st = s2
    if done: st = env.reset()
z = ag.world.encode(np.asarray([env.reset(2)]))
print("冒烟通过 logits=", np.round(ag.actor.forward(z)[0], 2))
