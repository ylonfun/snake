import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
os.environ["SDL_VIDEODRIVER"] = "dummy"
from rl_env import SnakeEnv
from dqn_agent import DQNAgent

for mode, make_env in [
    ("energy_move", lambda: SnakeEnv(energy_mode=True, allow_stay=False)),
    ("survival", lambda: SnakeEnv(survival_mode=True, init_len=8)),
]:
    env = make_env()
    agent = DQNAgent(state_mode=mode)
    st = env.reset(1)
    for _ in range(300):
        a = agent.act(st)
        s2, r, done, _ = env.step(a)
        agent.remember(st, a, r, s2, done)
        agent.learn()
        st = s2
        if done:
            st = env.reset()
    print(f"{mode}: 300步无维度错误，buffer状态列={agent.buffer.states.shape[1]} 网络输入={agent.q.W[0].shape[1]}")
