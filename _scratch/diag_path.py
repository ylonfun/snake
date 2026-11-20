import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint, CKPT_DIR

agent = DQNAgent()
load_checkpoint(agent, os.path.join(CKPT_DIR, "final.npz"))

seg_actual, seg_manh = [], []
for seed in range(20):
    env = SnakeEnv()
    state = env.reset(seed)
    prev_score, seg_steps = 0, 0
    head = env.body[0]
    pending_manh = abs(head[0] - env.food[0]) + abs(head[1] - env.food[1])
    while True:
        action = agent.act(state, training=False)
        state, _, done, info = env.step(action)
        seg_steps += 1
        if info["score"] > prev_score:
            seg_actual.append(seg_steps)
            seg_manh.append(pending_manh)
            prev_score, seg_steps = info["score"], 0
            if not done:
                h = env.body[0]
                pending_manh = abs(h[0] - env.food[0]) + abs(h[1] - env.food[1])
        if done:
            break

actual = np.asarray(seg_actual)
manh = np.asarray(seg_manh)
ratio = actual / manh
print(f"吃果段数={len(actual)}")
print(f"正好走理论最短(=曼哈顿距离)的比例={np.mean(actual == manh):.0%}")
print(f"平均实际步数 / 曼哈顿下限={ratio.mean():.2f}")
print(f"绕路最严重的段: 实际{actual[ratio.argmax()]}步 vs 最短{manh[ratio.argmax()]}步")
