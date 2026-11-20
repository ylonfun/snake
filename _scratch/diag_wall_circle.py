import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")

# 找一个能进入“长身绕圈”的局，记录最后 25 步的位置/气味/访问数
for seed in range(5000, 5020):
    env = SnakeEnv(); st = env.reset(seed)
    trace = []
    while True:
        a = ag.act(st, False)
        st2, r, done, info = env.step(a)
        hx, hy = env.body[0]
        on_edge = hx in (0, 9) or hy in (0, 9)
        trace.append((env.body[0], on_edge, float(env.scent[hy, hx]),
                      int(env.visit_count[hy, hx]), len(env.body)))
        if done:
            break
        st = st2
    if len(env.body) >= 8 and env.death_cause == "hunger":
        last = trace[-25:]
        edge_steps = sum(1 for t in last if t[1])
        print(f"seed {seed} 死因{env.death_cause} 最终体长{len(env.body)}")
        print(f"最后25步中在边界上的有 {edge_steps} 步")
        print("位置 / 是否边界 / 该格气味 / 访问数 / 体长（最后12步）")
        for t in last[-12:]:
            print("  ", t)
        break
