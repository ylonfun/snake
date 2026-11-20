import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint

agent = DQNAgent(state_mode="local")
load_checkpoint(agent, "dqn_ckpt/final.npz")

# 射线组索引：_VISION_DIRS 中 上=0 右=2 下=4 左=6
RAY = {"上": 0, "右": 2, "下": 4, "左": 6}
ACT_NAME = ["上", "下", "左", "右"]

for seed in [5000, 5002]:
    e = SnakeEnv(state_mode="local")
    st = e.reset(seed)
    while True:
        qvals = agent.q.forward(np.asarray([st], np.float32))[0]
        a = agent.act(st, training=False)
        st2, _, done, _ = e.step(a)
        if done:
            print(f"=== seed={seed}  死亡动作方向={ACT_NAME[a]} ===")
            print("  网络给4个动作的Q值:", np.round(qvals, 3),
                  "  选中动作Q=", round(qvals[a], 3))
            for name, gi in RAY.items():
                b = 8 + gi * 3
                inv_dis, see_food, see_self = st[b], st[b + 1], st[b + 2]
                print(f"  {name}方向射线: 1/dis到墙={inv_dis:.3f}  "
                      f"see_food={see_food:.0f}  see_self={see_self:.0f}")
            print()
            break
        st = st2
