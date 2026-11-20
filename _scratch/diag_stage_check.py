import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

STAGE = 4
agent = DreamerAgent(freeze_world=True, horizon=5)
agent.load_world("world_frozen.npz")
agent.load(os.path.join("dreamer_ckpt", f"stage_{STAGE}.npz"))

# ---- 1) 20 个随机局面：统计表现 + 是否成环 ----
env = SnakeEnv()
rows = []
for seed in range(7000, 7020):
    s = env.reset(seed)
    seen = {}
    looped = False
    while True:
        key = (tuple(env.body), env.food)
        if key in seen:
            looped = True
        seen[key] = True
        a = agent.act(s, training=False)
        s, r, done, info = env.step(a)
        if done:
            break
    rows.append((info["score"], info["steps"], env.death_cause, looped))

scores = [r[0] for r in rows]
steps = [r[1] for r in rows]
loops = sum(1 for r in rows if r[3])
causes = {}
for r in rows:
    causes[r[2]] = causes.get(r[2], 0) + 1
print(f"stage_{STAGE} 20局: 均分{np.mean(scores):.2f} 峰值{max(scores)} "
      f"平均步数{np.mean(steps):.0f} 成环局数{loops} 死因{causes}")

# ---- 2) 一个干净局面：logits 饱和程度 + 想象 Q 虚高 ----
s = env.reset(7000)
z = agent.world.encode(np.asarray([s], np.float32))
logits = agent.actor.forward(z)[0]
Q5 = agent._action_values(z, depth=5)
print("actor logits:", np.round(logits, 2), "最大绝对值", np.abs(logits).max())
print("想象Q(depth5):", np.round(Q5[0], 2), "峰值", np.abs(Q5).max())

# ---- 3) 想象 Q 是否随 depth 发散 ----
for d in (1, 2, 3, 5, 8, 10):
    Qd = agent._action_values(z, depth=d)
    print(f"depth={d:2d} Q峰值={np.abs(Qd).max():6.2f}")
