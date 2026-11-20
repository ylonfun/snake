"""最强规划器路径效率 + 兼容性检查。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent
from cnn_safe_agent import CNNMultiSpaceAgent

# 1) 兼容性：默认环境（growth=0）+ 实时多步空间
env0 = SnakeEnv(state_mode="grid")
s = env0.reset(1)
assert env0.deadline_growth == 0.0
ms = CNNMultiSpaceAgent(); ms.load(os.path.join("cnn_ckpt", "best.npz"))
s2, _, dn, _ = env0.step(ms.act(s, training=False, env=env0))
print("默认环境 growth=", env0.deadline_growth, "实时 agent 正常，done=", dn)

# 2) 最强离线规划器步数/路径效率（10 局）
agent = CNNBeamAgent(depth=10, beam=3, space_weight=8.0, tail_pen=5.0)
agent.load(os.path.join("cnn_ckpt", "best.npz"))
ratios = []
for seed in range(5000, 5010):
    env = SnakeEnv(state_mode="grid", deadline_growth=1.0)
    s = env.reset(seed); n = 0
    while True:
        a = agent.act(s, training=False, env=env)
        n += 1
        s, _, done, info = env.step(a)
        if done:
            eaten = info["score"]
            ratios.append(n/eaten)
            print(f"seed{seed} 吃{eaten} 步数{n} 每苹果={n/eaten:.1f} 死因={env.death_cause}")
            break
print("平均每苹果步数=", round(np.mean(ratios), 1))
