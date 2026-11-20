import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent

agent = DreamerAgent(freeze_world=True, horizon=5)
agent.load_world("world_frozen.npz")
agent.load(os.path.join("dreamer_ckpt", "stage_4.npz"))
w = agent.world

# 对多个 seed 的安全动作，对比"想象中 C"与"真实下一状态编码后的 C"
env = SnakeEnv()
imag_terms, real_terms = [], []
for seed in range(7000, 7010):
    s = env.reset(seed)
    z = w.encode(np.asarray([s], np.float32))
    for a in range(4):
        e2 = SnakeEnv(); e2.reset(seed)
        ns, r, done, info = e2.step(a)
        c_imag = w.C.forward(w.step(z, np.asarray([a])))[0, 0]
        if not done:  # 只看真实安全的动作
            c_real = w.C.forward(w.encode(np.asarray([ns], np.float32)))[0, 0]
            imag_terms.append(c_imag); real_terms.append(c_real)

imag_terms = np.array(imag_terms); real_terms = np.array(real_terms)
print(f"安全动作样本数: {len(imag_terms)}")
print(f"真实编码 C 均值={real_terms.mean():.3f} (应接近1=继续)")
print(f"想象预测 C 均值={imag_terms.mean():.3f}")
print(f"想象中误判为终止(C<0.5)的比例={(imag_terms < 0.5).mean():.3f}")
print(f"真实编码误判为终止的比例={(real_terms < 0.5).mean():.3f}")
