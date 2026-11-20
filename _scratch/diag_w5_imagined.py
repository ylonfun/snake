"""诊断想象规划：初始局面对四个动作的模型预测 vs 真实结果。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_imagined_agent import CNNImaginedAgent

agent = CNNImaginedAgent(lookahead=5)
agent.load(os.path.join("cnn_ckpt", "best.npz"))

for seed in [5000, 5001]:
    env = SnakeEnv(state_mode="grid")
    g = env.reset(seed)
    print("=" * 50, "初始头位置:", env.body[0])
    for a in range(4):
        # 真实
        snap = env.clone_state()
        _, r_real, done_real, _ = env.step(a)
        env.restore_state(snap)
        # 模型
        ng, pd = agent.world.forward(g[np.newaxis], [a])
        r_pred = agent._reward(g, ng[0], bool(pd[0] > .5))
        print("动作%d 真实(done=%s r=%.1f) 模型(done=%.3f r=%.2f)" % (
            a, done_real, r_real, pd[0], r_pred))
