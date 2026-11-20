import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from latent_world_model import LatentWorldModel

env = SnakeEnv()
st = env.reset(1)
print("状态维度:", st.shape[0])
wm = LatentWorldModel()
S, A, R, S2, D = [], [], [], [], []
for _ in range(100):
    a = np.random.randint(4)
    s2, r, done, _ = env.step(a)
    S.append(st); A.append(a); R.append(r); S2.append(s2); D.append(done); st = s2
    if done: st = env.reset()
wm.update_world(np.array(S, np.float32), np.array(A), np.array(R, np.float32),
                np.array(S2, np.float32), np.array(D, np.float32))
print("41维世界模型 update 通过，编码 z:", wm.encode(np.asarray([st])).shape)
