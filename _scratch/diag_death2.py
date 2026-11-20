import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint
from settings import DIRECTIONS

agent = DQNAgent(state_mode="local")
load_checkpoint(agent, "dqn_ckpt/final.npz")

wall = body = 0
death_lens = []
for seed in range(5000, 5040):
    e = SnakeEnv(state_mode="local")
    st = e.reset(seed)
    while True:
        a = agent.act(st, training=False)
        dx, dy = DIRECTIONS[a]
        hx, hy = e.body[0]
        nx, ny = hx + dx, hy + dy
        st, r, done, info = e.step(a)
        if done:
            death_lens.append(len(e.body))
            if not (0 <= nx < e.cols and 0 <= ny < e.rows):
                wall += 1
            else:
                body += 1
            break

print(f"40 局死亡原因：撞墙={wall}  撞自己身体={body}")
dl = np.array(death_lens)
print(f"死亡长度：平均={dl.mean():.1f} 最小={dl.min()} 最大={dl.max()}")
print(f"长度分布: <10={int((dl<10).sum())} 10-19={int(((dl>=10)&(dl<20)).sum())} "
      f"20-29={int(((dl>=20)&(dl<30)).sum())} >=30={int((dl>=30).sum())}")
