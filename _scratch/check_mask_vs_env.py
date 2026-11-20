import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv, DIRECTIONS
from dreamer_agent import DreamerAgent

ag = DreamerAgent(freeze_world=True)
ag.load_world("world_frozen.npz"); ag.load("dreamer_ckpt/latest.npz")


def env_legal(body, direction):
    """环境真实规则：每个动作执行后是否安全（含反向钳制、排除会移走的尾巴）。"""
    hard = set(body[:-1])
    out = []
    for act0 in range(4):
        nd = DIRECTIONS[act0]
        if nd == (-direction[0], -direction[1]):
            nd = direction
        h = (body[0][0] + nd[0], body[0][1] + nd[1])
        wall = not (0 <= h[0] < 10 and 0 <= h[1] < 10)
        self_hit = h in hard
        out.append(not (wall or self_hit))
    return out


env = SnakeEnv(); st = env.reset(5000)
for step in range(20):
    body, direction = env.body.copy(), env.direction
    mask = ag._legal_mask(st)
    real = env_legal(body, direction)
    head = body[0]
    if head[0] in (0, 9) or head[1] in (0, 9):
        print(f"步{step} 头{head} 方向{direction}")
        print("  mask:", np.array(mask, dtype=int), " 环境规则:", np.array(real, dtype=int),
              " 一致" if list(mask) == real else " *** 不一致 ***")
    a = ag.act(st, False)
    st, _, done, _ = env.step(a)
    if done:
        break
