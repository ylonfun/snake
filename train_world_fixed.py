"""W5a 固定数据集多轮训练：在可重复采样的数据上把动力学/奖励/继续头训到收敛。

在线训练时 buffer 持续增长、数据几乎不被重复，奖励头拟合稀有大量级事件需要约
上千 epoch，故先收集固定数据集（含足够吃苹果/死亡样本），再多轮训练并周期评估。
"""
import os, sys, time
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from latent_grid_model import LatentGridModel
from cnn_dqn_agent import CNNDQNAgent

N_DATA, BATCH, STEPS = 8000, 64, 30000


def collect():
    behavior = CNNDQNAgent(); behavior.load(os.path.join("cnn_ckpt", "best.npz"))
    env = SnakeEnv(state_mode="grid")
    G, A, R, G2, D = [], [], [], [], []
    s = env.reset()
    while len(G) < N_DATA:
        a = behavior.act(s, training=False)
        s2, r, done, _ = env.step(a)
        G.append(s); A.append(a); R.append(r); G2.append(s2); D.append(done)
        s = env.reset() if done else s2
    return (np.asarray(G, np.float32), np.asarray(A), np.asarray(R, np.float32),
            np.asarray(G2, np.float32), np.asarray(D))


def evaluate(model, G, A, R, G2, D):
    z2 = model.encode(G2)
    pr, pc = model.reward_of(z2), model.cont_prob(z2)
    food, death = R > 5, R < -5
    normal = ~food & ~death
    print("  奖励头 吃苹果=%.2f 死亡=%.2f 普通=%.3f" % (
        pr[food].mean(), pr[death].mean(), pr[normal].mean()))
    print("  继续头 死亡样本存活概率=%.3f 命中死亡=%d/%d" % (
        pc[D].mean(), ((pc < .5) & D).sum(), D.sum()))


def main():
    G, A, R, G2, D = collect()
    print("固定数据集:", G.shape, "吃苹果=%d 死亡=%d" % ((R > 5).sum(), D.sum()))
    model = LatentGridModel(z_dim=64)
    wall = time.time()
    for step in range(1, STEPS + 1):
        idx = np.random.randint(0, N_DATA, BATCH)
        model.update(G[idx], A[idx], R[idx], G2[idx], D[idx])
        if step % 5000 == 0:
            print("步 %d (约 %d epoch) 耗时=%ds" % (
                step, step * BATCH // N_DATA, time.time() - wall))
            evaluate(model, G, A, R, G2, D)
    model.save(os.path.join("cnn_ckpt", "world_cnn.npz"))
    print("已保存 cnn_ckpt/world_cnn.npz")


if __name__ == "__main__":
    main()
