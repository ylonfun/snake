import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv

env = SnakeEnv()
env.reset(3)
fx, fy = env.food
sc = env.scent

print("气味场（苹果在", (fx, fy), "）：")
for y in range(10):
    print("  " + " ".join(f"{sc[y,x]:.3f}" for x in range(10)))

print("\n--- 约束验证 ---")
print("苹果格气味 =", round(float(sc[fy, fx]), 4), "（应=0.015，且<移动成本0.02）")
print("全部格子气味>0（全局覆盖）：", bool((sc > 0).all()), " 最小气味 =", round(float(sc.min()), 6))
print("最浓普通步净奖励 = -0.02 + 0.015 =", round(-0.02 + float(sc.max()), 4), "（负，无法靠闻获利）")
print("重复访问第二圈净奖励 = -0.02+0.015-0.05 =", round(-0.02 + 0.015 - 0.05, 4), "（更负）")

# 梯度方向：每个格子朝苹果走一步，气味应增大
ok = True
for y in range(10):
    for x in range(10):
        if (x, y) == (fx, fy):
            continue
        sx = np.sign(fx - x); sy = np.sign(fy - y)
        # 只沿一个轴靠近
        nx, ny = (x + sx, y) if sx != 0 else (x, y + sy)
        if sc[ny, nx] <= sc[y, x]:
            ok = False
print("朝苹果方向移动气味必然增大（梯度指向苹果）：", ok)

# 端到端运行
st = env.reset(1)
for _ in range(200):
    st, r, done, _ = env.step(np.random.randint(4))
    if done:
        st = env.reset()
print("环境200步运行无报错")
