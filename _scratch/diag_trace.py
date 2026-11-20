import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent, softmax

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

env = SnakeEnv()
st = env.reset(5000)
cur = ag.world.encode(np.asarray([st]))
print("初始 actor logits:", np.round(ag.actor.forward(cur)[0], 2))
for t in range(10):
    lg = ag.actor.forward(cur)
    p = softmax(lg)
    a = int(p.argmax(1)[0])
    z_next = ag.world.step(cur, np.asarray([a]))
    r = ag.world.R.forward(z_next)[0, 0]
    c = ag.world.C.forward(z_next)[0, 0]
    v = ag.critic.forward(z_next)[0, 0]
    print(f"t{t} a={a} p={np.round(p[0],2)} r={r:.2f} cont={c:.2f} V_next={v:.2f}")
    cur = z_next
    if c < 0.5:
        print("想象终止"); break
