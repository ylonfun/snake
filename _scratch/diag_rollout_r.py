import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")
env = SnakeEnv(); st = env.reset(0)

# 走到 (2,6)
while env.body[0] != (2, 6):
    a = ag.act(st, False)
    st, _, done, _ = env.step(a)
    if done:
        break

z = ag.world.encode(np.asarray([st], np.float32))
# 选动作1（向下），手动展开 5 步，打印每步原始R预测、裁剪后奖励、隐状态是否偏离
cur = ag.world.step(z, np.asarray([1]))
for t in range(5):
    raw = float(ag.world.R.forward(cur)[0, 0])
    clipped = float(ag.world.reward(cur)[0])
    print(f"步{t+1} 原始R={raw:+.2f} 裁剪后={clipped:+.2f} 隐状态均值={float(cur.mean()):+.3f}")
    a2 = int(ag.actor.forward(cur).argmax(1)[0])
    cur = ag.world.step(cur, np.asarray([a2]))
