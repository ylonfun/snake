import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint
from mlp import MLP

beh = DQNAgent(state_mode="local"); load_checkpoint(beh, "dqn_ckpt/final.npz"); beh.eps = 0.3
env = SnakeEnv(); st = env.reset()
X, Y = [], []
while len(X) < 20000:
    a = beh.act(st[:40], True)
    oh = np.zeros(4, np.float32); oh[a] = 1
    X.append(np.concatenate([st, oh]))
    st2, r, done, _ = env.step(a)
    Y.append(1.0 if r == env.r_food else 0.0)
    st = env.reset() if done else st2
X = np.array(X, np.float32); Y = np.array(Y, np.float32)
cut = int(0.9 * len(X))
print("吃苹果事件占比 =", round(Y.mean(), 4))

clf = MLP([46, 64, 1], lr=1e-3, out_act="sigmoid")
for t in range(3000):
    idx = np.random.randint(0, cut, 128)
    p = clf.forward(X[idx])[:, 0]
    w = np.where(Y[idx] == 1, 20.0, 1.0)
    clf.backward(((p - Y[idx]) * w / len(idx)).reshape(-1, 1))
    clf.clip_grads(5.0); clf.adam_step()

pt = clf.forward(X[cut:])[:, 0]
yt = Y[cut:]
pred = (pt > 0.5).astype(np.float32)
food = yt == 1
print("整体准确率 =", round((pred == yt).mean(), 4))
print("吃苹果召回 =", round((pred[food] == 1).mean(), 4), f"（吃苹果样本{food.sum()}个）")
print("误报率 =", round((pred[~food] == 1).mean(), 4))
