"""检查模型预测的 next_grid 头位置：死亡动作头应不动，安全动作头应移动。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_imagined_agent import CNNImaginedAgent

agent = CNNImaginedAgent(lookahead=5)
agent.load(os.path.join("cnn_ckpt", "best.npz"))

for seed in [5000, 5001, 5002]:
    env = SnakeEnv(state_mode="grid")
    g = env.reset(seed)
    hx, hy = np.unravel_index(np.argmax(g[0]), (10, 10))
    print("=" * 50, "头:", (hx, hy))
    for a in range(4):
        snap = env.clone_state()
        _, _, done_real, _ = env.step(a)
        env.restore_state(snap)
        ng, _ = agent.world.forward(g[np.newaxis], [a])
        phx, phy = np.unravel_index(np.argmax(ng[0, 0]), (10, 10))
        moved = (phx, phy) != (hx, hy)
        print("动作%d 真实死亡=%s 预测头移动=%s 预测头位置=%s" % (
            a, done_real, moved, (phx, phy)))
