"""代码审查修复后的回归验证：GA 纪录回放 + clone_state energy + safe agent 尺寸。

1) 纪录基因在"循环检测已生效"的新 ai_game 下回放，分数必须与存档一致；
2) rl_env.clone_state 现在保存 energy，能量模式克隆/恢复后能量一致；
3) cnn_safe_agent 的 flood fill 归一化在 10×10 下与旧硬编码结果完全一致。
"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np

# ---- 1) 纪录回放：循环检测生效后分数不退化 ----
import ai_game
with open(os.path.join("genes", "record", "genes")) as f:
    genes = np.array(list(map(float, f.read().split())))
with open(os.path.join("genes", "record", "seed")) as f:
    seed = int(f.read())
with open(os.path.join("genes", "record", "score.txt")) as f:
    record_score = int(f.read())

score, steps, shaped, _ = ai_game.Game([genes], seed=seed, show=False).play()
print(f"纪录回放: score={score} steps={steps} (存档 {record_score} 分)")
assert score == record_score, f"回放分数退化: {score} != {record_score}"
print("  OK: 循环检测生效后纪录回放分数不变")

# ---- 2) clone_state 保存 energy ----
from rl_env import SnakeEnv
env = SnakeEnv(energy_mode=True, allow_stay=False)
env.reset(7)
env.energy = 42.5                      # 直接设哨兵值，避开碰撞/步数干扰
snap = env.clone_state()
env.energy = 1.0
env.restore_state(snap)
assert env.energy == 42.5, f"energy 未恢复: {env.energy} != 42.5"
print("  OK: clone/restore 后 energy=42.5 与快照一致")
# 旧快照（无 energy 键）兼容：恢复后不报错、保持当前值
legacy = {k: v for k, v in snap.items() if k != "energy"}
env.energy = 9.0
env.restore_state(legacy)
assert env.energy == 9.0
print("  OK: 旧格式快照（无 energy 键）向后兼容")

# ---- 3) cnn_safe_agent 归一化等价（10×10 下 cells=100 与旧 /100 相同）----
from cnn_safe_agent import CNNSafeAgent, flood_fill
agent = CNNSafeAgent()
env2 = SnakeEnv()
env2.reset(3)
s_new = agent._flood_space(env2, 0)
s_old = flood_fill(env2.body[0], set(env2.body[1:]))   # 旧默认 10×10
assert s_new == s_old
print(f"  OK: _flood_space={s_new} 与旧默认参数结果一致")

print("全部回归通过")
