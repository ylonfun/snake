import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv

# 收集大量局面，统计"相同局部观测"对应多少不同的真实全局状态
env = SnakeEnv()
obs_to_globals = {}
for seed in range(20000):
    s = env.reset(seed)
    key = tuple(np.round(s, 3))
    glob = (tuple(env.body), env.food, tuple(env.direction))
    obs_to_globals.setdefault(key, set()).add(glob)

n_obs = len(obs_to_globals)
ambiguous = {k: v for k, v in obs_to_globals.items() if len(v) > 1}
sizes = [len(v) for v in obs_to_globals.values()]
print(f"共 {20000} 个局面，去重后不同局部观测数 = {n_obs}")
print(f"其中对应多个真实局面的观测数 = {len(ambiguous)} ({len(ambiguous)/n_obs:.1%})")
print(f"单个观测最多对应 {max(sizes)} 个不同真实局面")
# 展示一个例子
if ambiguous:
    k = next(iter(ambiguous))
    print("\n示例：相同观测对应", len(ambiguous[k]), "个真实局面")
    for g in list(ambiguous[k])[:4]:
        print("  身体", g[0], "食物", g[1])
