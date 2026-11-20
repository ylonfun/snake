import os, sys

root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")

for seed in range(5000, 5040):
    env = SnakeEnv(); st = env.reset(seed)
    while True:
        a = ag.act(st, False)
        legal = ag._legal_mask(st)
        st2, _, done, _ = env.step(a)
        if done and env.death_cause == "wall":
            print("seed", seed, "致命动作", a, "mask判定该动作安全?", bool(legal[a]))
            for aa, ray in enumerate(ag._ACTION_RAY):
                b = 8 + ray * 4
                print(f"  动作{aa} 射线{ray} 1/dis={st[b]:.4f} adj={st[b+3]:.0f}")
            break
        st = st2
        if done:
            break
