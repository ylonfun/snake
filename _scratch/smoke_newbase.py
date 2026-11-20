import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz")
ag.load("dreamer_ckpt/latest.npz")
env = SnakeEnv(); st = env.reset(1)
for _ in range(200):
    st, r, done, info = env.step(ag.act(st, False))
    if done:
        st = env.reset()
print("新base端到端200步决策循环正常，当前得分", info["score"], "步数", info["steps"])
