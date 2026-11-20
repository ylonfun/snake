import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")
env = SnakeEnv(); st = env.reset(0)

seen = set()
while True:
    a = ag.act(st, False)
    key = env.body[0]
    if key in seen and key in {(1, 8), (1, 7), (1, 6), (2, 6), (2, 7), (2, 8)}:
        z = ag.world.encode(np.asarray([st], np.float32))
        logits = ag.actor.forward(z)[0]
        Q = ag._action_values(z)[0]
        hx, hy = key
        print(f"状态{key} 苹果{env.food} 当前气味{env.scent[hy,hx]:.4f}")
        for i, name in enumerate(["上", "下", "左", "右"]):
            print(f"  {name}: actor分{logits[i]:+.3f} 想象Q{Q[i]:+.3f}")
        print("  -> 实际选", ["上", "下", "左", "右"][a])
    seen.add(key)
    st2, _, done, _ = env.step(a)
    if done or len(seen) > 30:
        break
    st = st2
