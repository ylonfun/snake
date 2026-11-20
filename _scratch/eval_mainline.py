import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint

agent = DQNAgent(state_mode="local")
load_checkpoint(agent, "dqn_ckpt/final.npz")

scores, steps_all, death_len, ratios = [], [], [], []
exact = 0
max_seg, total_seg = 0, 0
for seed in range(5000, 5030):
    e = SnakeEnv(state_mode="local")
    st = e.reset(seed)
    seg_start = e.body[0]
    seg_steps = 0
    while True:
        a = agent.act(st, training=False)
        st, r, done, info = e.step(a)
        seg_steps += 1
        if r > 0:
            manh = abs(seg_start[0]-e.body[0][0]) + abs(seg_start[1]-e.body[0][1])
            if manh > 0:
                ratios.append(seg_steps/manh); total_seg += 1
                if seg_steps == manh:
                    exact += 1
            max_seg = max(max_seg, seg_steps)
            seg_start = e.body[0]; seg_steps = 0
        if done:
            scores.append(info["score"]); steps_all.append(e.steps)
            death_len.append(len(e.body))
            break

scores = np.array(scores)
print(f"评估 30 局（纯贪心）：")
print(f"单局最高分={scores.max()}  平均分={scores.mean():.1f}  中位数={np.median(scores):.0f}")
print(f"分数分布: 到30+={int((scores>=30).sum())}局  20-29={int(((scores>=20)&(scores<30)).sum())}局  "
      f"10-19={int(((scores>=10)&(scores<20)).sum())}局  <10={int((scores<10).sum())}局")
print(f"平均步数={np.mean(steps_all):.0f}  平均死亡长度={np.mean(death_len):.1f}")
print(f"路径比={np.mean(ratios):.2f}(1.00最短)  最短段占比={100*exact/total_seg:.0f}%  最长段={max_seg}")
