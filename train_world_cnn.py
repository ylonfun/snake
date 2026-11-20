"""W5a：在线训练 LatentGridModel 并验证多步想象的可靠性。

用已训练的 CNN-DQN(best) 作为行为策略收集 transition（覆盖长身局面），在线联合
训练潜在模型；周期测量自回归 rollout 的想象 z 与真实编码 z 的误差，确认长 horizon
有界、不漂移，再进入 W5b 的 actor-critic 策略学习。
"""
import os
import sys
import time

root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np

from rl_env import SnakeEnv
from replay_buffer import ReplayBuffer
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

GRID_FLAT, GRID_SHAPE, BATCH = 600, (-1, 6, 10, 10), 64
TRAIN_STEPS = 12000
HORIZON = 10


def rollout_error(model, behavior, n_episodes=8, horizon=HORIZON):
    """自回归想象 z 与真实编码 z 的逐 step MSE。"""
    errs = {h: [] for h in range(1, horizon + 1)}
    for ep in range(n_episodes):
        env = SnakeEnv(state_mode="grid")
        g = env.reset(6000 + ep)
        traj_g, traj_a = [g], []
        for _ in range(horizon):
            a = behavior.act(g, training=False)
            g, _, done, _ = env.step(a)
            traj_a.append(a); traj_g.append(g)
            if done:
                break
        real_z = [model.encode(np.asarray([x], np.float32)) for x in traj_g]
        z = real_z[0]
        for t in range(len(traj_a)):
            z = model.step_latent(z, [traj_a[t]])
            errs[t + 1].append(float(np.mean((z - real_z[t + 1]) ** 2)))
    return {h: float(np.mean(v)) for h, v in errs.items() if v}


def main():
    behavior = CNNDQNAgent()
    behavior.load(os.path.join("cnn_ckpt", "best.npz"))
    model = LatentGridModel(z_dim=64)
    buf = ReplayBuffer(100000, GRID_FLAT)
    env = SnakeEnv(state_mode="grid")
    state = env.reset()
    wall = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        a = behavior.act(state, training=False)
        s2, r, done, _ = env.step(a)
        buf.push(state.reshape(-1), a, r, s2.reshape(-1), done)
        state = env.reset() if done else s2
        if buf.size >= BATCH:
            s, ac, rw, s2n, dn = buf.sample(BATCH)
            model.update(s.reshape(GRID_SHAPE), ac, rw, s2n.reshape(GRID_SHAPE), dn)
        if step % 2000 == 0:
            er = rollout_error(model, behavior)
            line = " ".join(f"h{h}={v:.4f}" for h, v in er.items())
            print(f"训练步 {step} 多步想象误差: {line} 耗时={time.time()-wall:.0f}s")

    model.save(os.path.join("cnn_ckpt", "world_cnn.npz"))
    print("模型已保存到 cnn_ckpt/world_cnn.npz")


if __name__ == "__main__":
    main()
