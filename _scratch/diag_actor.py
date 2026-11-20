import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from rl_env import SnakeEnv
from dreamer_agent import DreamerAgent, softmax

ag = DreamerAgent()
ag.load("dreamer_ckpt/latest.npz")
print("step_count=", ag.step_count)

for seed in range(5000, 5004):
    env = SnakeEnv()
    st = env.reset(seed)
    z = ag.world.encode(np.asarray([st]))
    lg = ag.actor.forward(z)
    p = softmax(lg)[0]
    v = ag.critic.forward(z)[0, 0]
    # 各动作是否致命
    fatal = []
    for a in range(4):
        e2 = SnakeEnv()
        e2.reset(seed)
        _, _, done, _ = e2.step(a)
        fatal.append(done)
    print(f"seed{seed} logits={np.round(lg[0],2)} p={np.round(p,2)} "
          f"argmax={lg.argmax(1)[0]} 致命={fatal} V={v:.2f}")
