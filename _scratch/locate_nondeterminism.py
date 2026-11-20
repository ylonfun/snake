"""同进程内两条独立轨迹逐动作对比，定位非确定性首个分歧点。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent


def make():
    ag = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
    env = SnakeEnv(rows=15, cols=15, state_mode="grid", deadline_growth=1.0)
    s = env.reset(5000)
    return ag, env, s


ag1, e1, s1 = make()
ag2, e2, s2 = make()
for step in range(2000):
    a1 = ag1.act(s1, training=False, env=e1)
    a2 = ag2.act(s2, training=False, env=e2)
    if a1 != a2:
        print(f"首个分歧 step{step}: 轨迹1动作{a1} 轨迹2动作{a2}")
        print("e1 头/食物/体长:", e1.body[0], e1.food, len(e1.body))
        print("e2 头/食物/体长:", e2.body[0], e2.food, len(e2.body))
        break
    s1, _, d1, _ = e1.step(a1)
    s2, _, d2, _ = e2.step(a2)
    if d1 or d2:
        print("某条结束", d1, d2, e1.death_cause, e2.death_cause)
        break
else:
    print("2000 步内无分歧")
