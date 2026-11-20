import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

from rl_env import SnakeEnv

# 1) 基本配置
env = SnakeEnv(energy_mode=True)
assert env.state_dim == 33 and env.n_actions == 5
s = env.reset(0)
assert len(s) == 33
print("能量模式状态维度=33, 动作数=5, 初始能量=", env.energy)

# 2) stay 消耗慢、移动消耗快
env.reset(0)
e0 = env.energy
env._step_energy(4)                 # stay
stay_cost = e0 - env.energy
e1 = env.energy
env._step_energy(0)                 # 移动
move_cost = e1 - env.energy
print(f"静止消耗={stay_cost:.3f} 移动消耗={move_cost:.3f}")
assert stay_cost < move_cost

# 3) 能量耗尽会饿死
env.reset(0)
env.energy = 0.001
_, r, done, info = env._step_energy(4)
assert done and info["score"] is not None
print("能量归零 -> 饿死 done=True")

# 4) 普通模式未被破坏
plain = SnakeEnv()
assert plain.state_dim == 32 and plain.n_actions == 4
plain.reset(0)
plain.step(0)
print("普通 local 模式仍正常: state_dim=32, n_actions=4")
print("能量环境冒烟测试通过")
