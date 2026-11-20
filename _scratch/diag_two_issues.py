import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")

for seed in range(5000, 5006):
    env = SnakeEnv(); st = env.reset(seed)
    after = []   # 最后一次吃苹果之后的记录
    eaten = 0
    while True:
        a = ag.act(st, False)
        st2, r, done, info = env.step(a)
        if r == env.r_food:
            eaten += 1; after = []      # 吃苹果就清空，只保留最后一次之后
        else:
            hx, hy = env.body[0]
            on_edge = hx in (0, 9) or hy in (0, 9)
            after.append((on_edge, float(env.scent[hy, hx])))
        if done:
            break
        st = st2
    if after:
        edge_ratio = np.mean([t[0] for t in after])
        mean_scent = np.mean([t[1] for t in after])
        print(f"seed{seed} 最后吃果时已吃{eaten}个/体长{eaten+1}；之后又走{len(after)}步，"
              f"边界占比{edge_ratio:.0%}，平均气味{mean_scent:.4f}，死因{env.death_cause}")
