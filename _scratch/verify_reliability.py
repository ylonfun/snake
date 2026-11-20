import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")
print("当前 horizon =", ag.horizon)
env = SnakeEnv(); st = env.reset(0)

max_q = -1e9; seen = set()
while True:
    key = env.body[0]
    if key not in seen:
        z = ag.world.encode(np.asarray([st], np.float32))
        Q = ag._action_values(z)[0]
        max_q = max(max_q, Q.max())
        if key == (2, 6):
            print("状态(2,6) 四动作想象Q =", np.round(Q, 2))
    seen.add(key)
    a = ag.act(st, False)
    st2, _, done, _ = env.step(a)
    if done or len(seen) > 40:
        break
    st = st2
print("整段想象 Q 的最大值 =", round(float(max_q), 2), "（修复前曾达 69；现在应被限制在有界范围）")
