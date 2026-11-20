"""固定状态下重复测量各首动作的 beam 值，定位不稳定环节。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

ag = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
env = SnakeEnv(rows=15, cols=15, state_mode="grid", deadline_growth=1.0)
s = env.reset(5000)
# 推进到 step4 状态
for _ in range(4):
    a = ag.act(s, training=False, env=env)
    s, _, done, _ = env.step(a)
print("固定状态 头=", env.body[0], "食物=", env.food, "体长=", len(env.body))

for rep in range(6):
    vals = [ag._beam_value(env, a) for a in range(4)]
    print(f"rep{rep} 各动作beam值=", [round(v, 6) for v in vals],
          "选择=", int(__import__("numpy").argmax(vals)))
