import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from dqn_agent import DQNAgent
from train_rl import load_checkpoint
from settings import DIRECTIONS

new = DQNAgent(state_mode="local")
load_checkpoint(new, "dqn_ckpt/final.npz")
old = DQNAgent(state_mode="local", net_sizes=(32, 64, 64, 4), n_actions=4)
load_checkpoint(old, "dqn_ckpt_32dim/final.npz")


def to32(s40):
    """40 维去掉每方向新增的 adj_self，还原为旧 32 维。"""
    out = list(s40[:8])
    for g in range(8):
        b = 8 + g * 4
        out += list(s40[b:b + 3])
    return np.array(out, np.float32)


def run(agent, seed, downgrade=False):
    e = SnakeEnv()
    st = e.reset(seed)
    score = steps = 0
    wall = body = 0
    ratios, exact, max_seg, nseg = [], 0, 0, 0
    seg_start, seg_steps = e.body[0], 0
    misjudge = 0          # 紧邻硬身体却选了该方向的次数
    while True:
        hx, hy = e.body[0]
        hard = set(e.body[:-1])
        agent_in = to32(st) if downgrade else st
        a = agent.act(agent_in, training=False)
        dx, dy = DIRECTIONS[a]
        if (hx + dx, hy + dy) in hard:
            misjudge += 1
        st, r, done, info = e.step(a)
        seg_steps += 1
        if r > 0:
            manh = abs(seg_start[0]-e.body[0][0]) + abs(seg_start[1]-e.body[0][1])
            if manh > 0:
                ratios.append(seg_steps/manh); nseg += 1
                if seg_steps == manh:
                    exact += 1
            max_seg = max(max_seg, seg_steps)
            seg_start, seg_steps = e.body[0], 0
        if done:
            nx, ny = hx+dx, hy+dy
            if not (0 <= nx < e.cols and 0 <= ny < e.rows):
                wall = 1
            else:
                body = 1
            score, steps = info["score"], info["steps"]
            return (score, steps, len(e.body), wall, body,
                    np.mean(ratios), 100*exact/max(nseg, 1), max_seg, misjudge)


for label, agent, dg in [("旧32维", old, True), ("新40维", new, False)]:
    R = np.array([run(agent, s, dg) for s in range(5000, 5030)])
    print(f"=== {label} ===")
    print(f"  均分={R[:,0].mean():.1f} 最高={R[:,0].max()} 平均步数={R[:,1].mean():.0f}")
    print(f"  死亡长度={R[:,2].mean():.1f} 撞墙={int(R[:,3].sum())} 撞身={int(R[:,4].sum())}")
    print(f"  路径比={R[:,5].mean():.2f} 最短段={R[:,6].mean():.0f}% 最长段={R[:,7].mean():.0f}")
    print(f"  紧邻身体仍选撞(误判)总次数={int(R[:,8].sum())}")
