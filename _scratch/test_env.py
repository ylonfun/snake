import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv
from ai_game import Snake
from settings import DIRECTIONS

# 1. 维度、dtype、随机 rollout 合法性、info 分数一致
env = SnakeEnv()
for seed in range(30):
    s = env.reset(seed)
    assert s.shape == (32,) and s.dtype == np.float32
    while True:
        s2, r, done, info = env.step(np.random.randint(4))
        assert s2.shape == (32,)
        if done:
            assert info["score"] == len(env.body) - 1
            break

# 2. 奖励规则：普通步/吃果/死亡
env.reset(1)
env.food = (env.body[0][0], env.body[0][1])  # 强制下一步吃到（朝食物方向）
# 直接构造：把头旁边设为食物
h = env.body[0]
for d in DIRECTIONS:
    cand = (h[0] + d[0], h[1] + d[1])
    if 0 <= cand[0] < 10 and 0 <= cand[1] < 10:
        env.food = cand
        env.direction = d
        before = len(env.body)
        _, r, done, info = env.step(DIRECTIONS.index(d))
        assert r == 10.0 and len(env.body) == before + 1
        break

# 3. 与 ai_game.Snake.get_state 在相同局面下输出一致
mismatch = 0
checked = 0
for seed in range(60):
    env.reset(seed)
    for _ in range(np.random.randint(0, 8)):
        _, _, done, _ = env.step(np.random.randint(4))
        if done:
            break
    if env.done:
        continue
    sk = Snake(env.body[0], env.direction,
               np.random.randn(964).astype(np.float32), env.cols, env.rows)
    sk.body = list(env.body)
    if not np.allclose(sk.get_state(env.food), env.get_state(), atol=1e-6):
        mismatch += 1
    checked += 1
print(f"状态一致性：检查 {checked} 局，不一致 {mismatch} 个")
print("全部测试通过")
