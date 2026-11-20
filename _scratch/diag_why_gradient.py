import os, sys, copy
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")

env = SnakeEnv(); st = env.reset(7)
shown = 0
for _ in range(400):
    a = ag.act(st, False)
    legal = ag._legal_mask(st)
    hx, hy = env.body[0]
    cur_scent = env.scent[hy, hx]
    rows = []
    for act_i in range(4):
        # 真实环境模拟该动作
        e2 = copy.deepcopy(env)
        _, r_real, done_real, _ = e2.step(act_i)
        nhx, nhy = e2.body[0]
        next_scent = float(e2.scent[nhy, nhx])
        # 世界模型预测
        z = ag.world.encode(np.asarray([st], np.float32))
        z2 = ag.world.step(z, np.asarray([act_i]))
        r_pred = float(ag.world.R.forward(z2)[0, 0])
        rows.append((act_i, legal[act_i], next_scent - cur_scent, r_pred, done_real))
    if shown < 4 and sum(legal) >= 2:
        z = ag.world.encode(np.asarray([st], np.float32))
        Q = ag._action_values(z)[0]
        print(f"状态 头{env.body[0]} 苹果{env.food} 当前气味{cur_scent:.4f} 实际选{a}")
        print("  动作 合法 真实气味增量 世界模型预测奖励 真实是否结束 | 想象Q")
        for i, r in enumerate(rows):
            print(f"   {r[0]}   {int(r[1])}    {r[2]:+.4f}      {r[3]:+.3f}        {int(r[4])}      | {Q[i]:+.3f}")
        shown += 1
    st2, _, done, _ = env.step(a)
    if done:
        st = env.reset()
    else:
        st = st2
