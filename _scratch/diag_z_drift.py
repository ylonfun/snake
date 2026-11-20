import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent
from dqn_agent import DQNAgent
from train_rl import load_checkpoint

ag = DreamerAgent(freeze_world=True); ag.load_world("world_frozen.npz")
beh = DQNAgent(state_mode="local"); load_checkpoint(beh, "dqn_ckpt/final.npz"); beh.eps = 0.3

H = 10
drift = [[] for _ in range(H + 1)]
for seed in range(6000, 6040):
    env = SnakeEnv(); c = env.reset(seed)
    states, actions = [c], []
    for _ in range(H):
        a = beh.act(c[:40], False)
        c, _, done, _ = env.step(a)
        states.append(c); actions.append(a)
        if done: break
    if len(actions) < H: continue
    # 自回归想象
    z_im = ag.world.rollout(ag.world.encode(np.asarray([states[0]]))[0], actions)
    for t in range(H + 1):
        z_real = ag.world.encode(np.asarray([states[t]]))[0]
        drift[t].append(np.abs(z_im[t] - z_real).mean())
print("隐状态 z 漂移（想象 vs 真实，平均绝对差）：")
for t in range(H + 1):
    print(f"  步{t}: {np.mean(drift[t]):.4f}")
