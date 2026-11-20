"""干净复现 seed5000（15x15 depth10 beam2），确认结果。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent

agent = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
env = SnakeEnv(rows=15, cols=15, state_mode="grid", deadline_growth=1.0)
s = env.reset(5000)
print("初始食物=", env.food, "初始方向=", env.direction)
nsteps = 0
while True:
    a = agent.act(s, training=False, env=env)
    nsteps += 1
    s, _, done, info = env.step(a)
    if nsteps % 500 == 0:
        print(f"step{nsteps} 吃{info['score']} 体长{len(env.body)}")
    if done:
        print(f"结束 吃{info['score']} 步数{nsteps} 死因={env.death_cause}")
        break
