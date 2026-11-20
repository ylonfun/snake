"""诊断 beam search 死亡原因。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from rl_env import SnakeEnv
from cnn_beam_agent import CNNBeamAgent
from cnn_safe_agent import flood_fill

agent = CNNBeamAgent(depth=10, beam=3, space_weight=8.0)
agent.load(os.path.join("cnn_ckpt", "best.npz"))
causes = {}
for seed in range(5000, 5015):
    env = SnakeEnv(state_mode="grid")
    s = env.reset(seed)
    while True:
        a = agent.act(s, training=False, env=env)
        # 检查被选动作执行前，是否存在安全动作
        snap = env.clone_state()
        safe_actions = []
        for aa in range(4):
            env.restore_state(snap)
            _, _, dn, _ = env.step(aa)
            if not dn:
                safe_actions.append(aa)
        env.restore_state(snap)
        s, _, done, info = env.step(a)
        if done:
            c = env.death_cause
            causes[c] = causes.get(c, 0) + 1
            sp = flood_fill(env.body[0], set(env.body[1:]))
            print(f"seed{seed} 分{info['score']} 死因={c} 死亡时连通={sp} "
                  f"死前可选安全动作={safe_actions} 选了={a}")
            break
print("死因分布：", causes)
