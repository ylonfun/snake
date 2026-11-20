"""确定性奖励函数：从 (grid, next_grid, done) 推断奖励，并验证与真实奖励一致性。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_dqn_agent import CNNDQNAgent

behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
env = SnakeEnv(state_mode="grid")


def reward_from(grid, next_grid, done):
    """done=-10；新头进入当前食物格=+10；普通步=-0.02+新头气味。"""
    if done:
        return -10.0
    fy, fx = np.unravel_index(np.argmax(grid[2]), (10, 10))
    hy, hx = np.unravel_index(np.argmax(next_grid[0]), (10, 10))
    if (hy, hx) == (fy, fx):
        return 10.0
    return -0.02 + float(next_grid[3, hy, hx])


G, R, G2, D = [], [], [], []
s = env.reset(5000)
while len(G) < 3000:
    a = behavior.act(s, training=False)
    s2, r, done, _ = env.step(a)
    G.append(s); R.append(r); G2.append(s2); D.append(done)
    s = env.reset(5000 + len(G)) if done else s2
R = np.asarray(R); D = np.asarray(D)
pred = np.array([reward_from(G[i], G2[i], D[i]) for i in range(len(R))])

food, death = R > 5, R < -5
print("吃苹果 推断=%.2f 真实=%.2f n=%d" % (pred[food].mean(), R[food].mean(), food.sum()))
print("死亡   推断=%.2f 真实=%.2f n=%d" % (pred[death].mean(), R[death].mean(), death.sum()))
print("普通步 推断=%.3f 真实=%.3f" % (pred[~food & ~death].mean(),
                                  R[~food & ~death].mean()))
print("吃苹果判定完全一致:", np.array_equal(food, pred > 5))
