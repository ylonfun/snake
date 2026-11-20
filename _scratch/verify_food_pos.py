import os, sys

root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
from rl_env import SnakeEnv


def collect(make_env, n):
    env = make_env(); st = env.reset(1)
    foods = set()
    for _ in range(n):
        st, _, done, _ = env.step(env.action_space_sample() if hasattr(env, "action_space_sample") else 0)
        foods.add(env.food)
        if done:
            st = env.reset()
    return foods


# 用随机动作跑，过程中记录所有苹果位置
import random as _r
rng = _r.Random(0)
for name, maker in [
    ("标准", lambda: SnakeEnv()),
    ("能量", lambda: SnakeEnv(energy_mode=True, allow_stay=False)),
    ("生存", lambda: SnakeEnv(survival_mode=True)),
]:
    env = maker(); env.reset(2)
    foods = set()
    for _ in range(2000):
        a = rng.randrange(env.n_actions)
        env.step(a)
        foods.add(env.food)
        if env.done:
            env.reset()
    on_edge = [f for f in foods if f[0] in (0, 9) or f[1] in (0, 9)]
    print(f"{name}: 记录到{len(foods)}个不同苹果位置，落在边界的有 {len(on_edge)} 个")
