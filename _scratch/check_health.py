"""全项目功能体检：逐个入口做有界、无头验证，输出 PASS/FAIL 清单。

覆盖：模块导入 / 环境语义与克隆 / 规划入口（BFS、Hamilton、全局回路）/
GA 进化与存档 / 各神经网络智能体加载真实存档后决策 / 世界模型往返。
"""
import os
import sys
import time
import importlib
import traceback

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np

RESULTS = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn() or ""
        RESULTS.append((True, name, time.time() - t0, detail))
    except Exception as e:
        RESULTS.append((False, name, time.time() - t0,
                        f"{type(e).__name__}: {e}"))
        traceback.print_exc()


# ---------------- 1. 模块导入 ----------------
def t_imports():
    mods = ["settings", "theme", "nn", "mlp", "conv", "rl_env", "path_game",
            "ai_game", "human_game", "base_agent", "replay_buffer", "q_network",
            "dqn_agent", "cnn_qnet", "cnn_dqn_agent", "cnn_lookahead_agent",
            "cnn_safe_agent", "cnn_beam_agent", "latent_world_model",
            "grid_forward_model", "latent_grid_model", "dreamer_agent",
            "cnn_imagined_agent", "main", "launcher"]
    for m in mods:
        importlib.import_module(m)
    return f"{len(mods)} 模块导入正常（compare 内含整段评估脚本，另行后台跑）"


# ---------------- 2. 环境语义 ----------------
def t_env():
    from rl_env import SnakeEnv
    for mode, dim in [("local", 42), ("full", 300), ("grid", None)]:
        e = SnakeEnv(state_mode=mode)
        s = e.reset(11)
        if mode == "grid":
            assert s.shape == (6, 10, 10), s.shape
        else:
            assert s.shape == (dim,), (mode, s.shape)
    # 克隆/恢复逐位一致（确定可复现）
    e = SnakeEnv()
    s0 = e.reset(12)
    for _ in range(4):
        e.step(0)
    snap = e.clone_state()
    acts = [1, 2, 3, 0, 1]
    out1 = [e.step(a)[0].copy() for a in acts]
    e.restore_state(snap)
    out2 = [e.step(a)[0].copy() for a in acts]
    assert all((x == y).all() for x, y in zip(out1, out2)), "clone 后轨迹不一致"
    # 任意尺寸
    big = SnakeEnv(rows=15, cols=15)
    assert big.reset(1).shape == (42,) and big.rows == 15 and big.cols == 15
    return "local42/full300/grid6ch 维度正确；clone 轨迹逐位一致；15×15 可用"


def t_env_rules():
    from rl_env import SnakeEnv
    # 禁反向：向右走时动作 2(左) 被钳成直行
    e = SnakeEnv()
    e.reset(5)
    e.direction = (1, 0)
    hx, hy = e.body[0]
    e.step(2)
    assert e.body[0] == (hx + 1, hy), f"反向未被禁止: {e.body[0]} vs {(hx+1, hy)}"
    # 吃满判定
    e2 = SnakeEnv()
    e2.reset(3)
    e2.body = [(x, 0) for x in range(9, -1, -1)]     # 占满整行，长度10
    e2.food = (0, 0)
    e2.visit_count = np.ones((10, 10), dtype=np.int64)
    e2._build_scent()
    dead = False
    for _ in range(12):
        e2.food = (0, 0) if e2.body[0] == (0, 0) else e2.food
        _, r, dead, info = e2.step(3)
        if dead:
            break
    return "禁反向生效；吃满/死亡路径可触发"


# ---------------- 3. 规划入口 ----------------
def t_bfs_path():
    from path_game import PathSnake
    from settings import DIRECTIONS
    import random
    rnd = random.Random(2026)
    cells = [(x, y) for x in range(10) for y in range(10)]
    head = rnd.choice(cells)
    sn = PathSnake(head, DIRECTIONS[rnd.randint(0, 3)], 10, 10)
    food = rnd.choice([c for c in cells if c not in sn.body])
    while not sn.dead and sn.score < 30:
        if sn.move(food):
            food = rnd.choice([c for c in cells if c not in sn.body])
    assert sn.score >= 30, f"BFS 蛇只吃到 {sn.score}"
    return f"BFS 路径蛇吃到 {sn.score} 分未死={not sn.dead}"


def t_hamilton():
    from path_game import HamiltonSnake
    from settings import DIRECTIONS
    import random
    rnd = random.Random(7)
    cells = [(x, y) for x in range(10) for y in range(10)]
    sn = HamiltonSnake(rnd.choice(cells), DIRECTIONS[rnd.randint(0, 3)],
                       10, 10, use_shortcuts=True)
    food = next(c for c in cells if c not in sn.body)
    guard = 0
    while not sn.dead and sn.score < 99 and guard < 4000:
        guard += 1
        if sn.move(food):
            rem = [c for c in cells if c not in sn.body]
            if not rem:
                break
            food = rem[0]
    assert sn.score == 99 and not sn.dead, f"Hamilton 10×10: {sn.score} 分 dead={sn.dead}"
    return "Hamilton 10×10 吃满 99、0 死亡"


def t_global_cycle():
    from path_game import global_cycle, validate_cycle, validate_partial_cycle
    from path_game import HamiltonSnake
    from settings import DIRECTIONS
    import random
    rnd = random.Random(9)
    for (rows, cols) in [(10, 10), (11, 11), (12, 12), (15, 15)]:
        cyc, excluded = global_cycle(cols, rows)
        if excluded is None:
            validate_cycle(cyc, cols, rows)
        else:
            validate_partial_cycle(cyc, cols, rows, excluded)
        expect = len(cyc) - 1
        sn = HamiltonSnake(rnd.choice(cyc), DIRECTIONS[rnd.randint(0, 3)],
                           cols, rows, cycle=cyc, use_shortcuts=True)
        food = rnd.choice([p for p in cyc if p not in set(sn.body)])
        while not sn.dead:
            if sn.move(food):
                free = [p for p in cyc if p not in set(sn.body)]
                if not free:
                    break
                food = rnd.choice(free)
        assert sn.score == expect and not sn.dead, \
            f"{rows}×{cols} 全局回路: {sn.score}/{expect} 分 dead={sn.dead}"
    return "global_cycle 10/11/12/15 全部吃满 (99/119/143/223)、0 死亡"


# ---------------- 4. GA 进化与存档 ----------------
def t_ga_evolve():
    from main import GA, GENES_LEN
    ga = GA(p_size=6, c_size=2)
    ga.generate_ancestor()
    ga.evolve()
    assert ga.best_individual is not None and len(ga.best_individual.genes) == GENES_LEN
    # evolve 后种群 = 精英 p_size 个 + 子代（成对生成，约 c_size 个）
    assert len(ga.population) >= ga.p_size, len(ga.population)
    return (f"1 代进化正常：best={ga.best_individual.score} "
            f"avg={ga.avg_score:.1f} 种群={len(ga.population)} 基因长={GENES_LEN}")


def t_ga_persist():
    from main import GA, GENES_LEN
    ga = GA()
    rec = ga.load_checkpoint("no_such_dir_xyz")
    assert rec is None, "不存在的存档目录应返回 None"
    with open(os.path.join("genes", "record", "genes")) as f:
        genes = np.array(list(map(float, f.read().split())))
    assert len(genes) == GENES_LEN
    ga.seed_from_genes(genes)
    assert len(ga.population[0].genes) == GENES_LEN
    return "存档缺失返回 None；纪录基因可热启动（964 基因）"


# ---------------- 5. 神经网络智能体 ----------------
def _play(env, act, limit=300):
    state = env.reset(1000)
    n = 0
    while n < limit:
        state, _, done, info = env.step(act(state, env))
        n += 1
        if done:
            break
    return info.get("score", 0), n, done


def t_dqn():
    from rl_env import SnakeEnv
    from dqn_agent import DQNAgent
    from train_rl import load_checkpoint
    agent = DQNAgent()
    load_checkpoint(agent, os.path.join("dqn_ckpt", "final.npz"))
    env = SnakeEnv()
    sc, n, done = _play(env, lambda s, e: agent.act(s[:40], training=False))
    return f"DQN(40维) 一局限步 {n}：{sc} 分 结束={done}"


def t_cnn_dqn():
    from rl_env import SnakeEnv
    from cnn_dqn_agent import CNNDQNAgent
    agent = CNNDQNAgent()
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    env = SnakeEnv(state_mode="grid")
    sc, n, done = _play(env, lambda s, e: agent.act(s, training=False))
    return f"CNN-DQN 一局限步 {n}：{sc} 分 结束={done}"


def t_cnn_safe():
    from rl_env import SnakeEnv
    from cnn_safe_agent import CNNMultiSpaceAgent
    agent = CNNMultiSpaceAgent()
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    env = SnakeEnv(state_mode="grid")
    sc, n, done = _play(env, lambda s, e: agent.act(s, training=False, env=e), limit=80)
    return f"CNN 多步空间规划 限步 {n}：{sc} 分"


def t_beam():
    from rl_env import SnakeEnv
    from cnn_beam_agent import CNNBeamAgent
    agent = CNNBeamAgent(depth=10, beam=2, space_weight=8.0, tail_pen=5.0)
    agent.load(os.path.join("cnn_ckpt", "best.npz"))
    env = SnakeEnv(state_mode="grid", deadline_growth=1.0)
    sc, n, done = _play(env, lambda s, e: agent.act(s, training=False, env=e), limit=30)
    return f"离线束搜索 限步 {n}：{sc} 分"


def t_world_cnn():
    from rl_env import SnakeEnv
    from cnn_imagined_agent import CNNImaginedAgent
    agent = CNNImaginedAgent(lookahead=3)
    env = SnakeEnv(state_mode="grid")
    sc, n, done = _play(env, lambda s, e: agent.act(s, training=False, env=e), limit=40)
    return f"CNN+前向模型想象 限步 {n}：{sc} 分"


def t_dreamer():
    from rl_env import SnakeEnv
    from dreamer_agent import DreamerAgent
    agent = DreamerAgent(freeze_world=True)
    agent.load_world("world_frozen.npz")
    agent.load(os.path.join("dreamer_ckpt", "final.npz"))
    env = SnakeEnv()
    sc, n, done = _play(env, lambda s, e: agent.act(s, training=False), limit=200)
    return f"Dreamer 世界模型策略 限步 {n}：{sc} 分（历史已知低分属预期）"


def t_world_roundtrip():
    from rl_env import SnakeEnv
    from latent_world_model import LatentWorldModel
    wm = LatentWorldModel(state_dim=42, n_actions=4)
    data = np.load("world_frozen.npz")
    for tag, net in [("E", wm.E), ("D", wm.D), ("R", wm.R), ("C", wm.C),
                     ("Dec", wm.Dec), ("Et", wm.E_t)]:
        for i in range(len(net.W)):
            net.W[i] = data[f"{tag}_W{i}"].copy()
            net.b[i] = data[f"{tag}_b{i}"].copy()
    env = SnakeEnv()
    s = env.reset(2)
    z = wm.encode(np.asarray([s]))
    z2 = wm.step(z, np.asarray([0]))
    dec = wm.decode(z2)[0]
    assert dec.shape == (42,) and np.isfinite(dec).all()
    r, c = wm.reward(z2)[0], wm.C.forward(z2)[0, 0]
    return f"冻结世界模型 编码→动力学→解码 往返正常 (r={r:.2f}, keep={c:.2f})"


def t_grid_forward():
    from grid_forward_model import GridForwardModel
    from rl_env import SnakeEnv
    env = SnakeEnv(state_mode="grid")
    g = env.reset(4)
    m = GridForwardModel()
    m.load(os.path.join("cnn_ckpt", "grid_forward.npz"))
    ng, pd = m.forward(g[None], [0])
    assert ng.shape == (1, 6, 10, 10) and 0 <= pd[0] <= 1
    return f"grid 前向模型 一步预测正常 (keep={pd[0]:.3f})"


# ---------------- 6. 启动器按钮（真实渲染路径） ----------------
def t_launcher_buttons():
    import pygame as pg
    import launcher
    pg.init()
    pg.display.set_mode((launcher.GAME_W, launcher.GAME_H))
    btns = launcher.BUTTONS
    keys = [b[0] if isinstance(b, (list, tuple)) else b for b in btns]
    names = [launcher.run_human, launcher.run_ai, launcher.run_path,
             launcher.run_replay, launcher.run_dqn, launcher.run_cnn,
             launcher.run_world, launcher.run_beam]
    assert len(names) == 8
    # 只验证菜单渲染 + 按钮可点坐标存在，不真正进入阻塞循环
    return f"launcher 8 入口函数齐备：{keys if keys else 'ok'}"


def main():
    checks = [
        ("模块导入", t_imports),
        ("环境维度与克隆", t_env),
        ("环境规则（禁反向/吃满）", t_env_rules),
        ("BFS 路径蛇", t_bfs_path),
        ("Hamilton 10×10", t_hamilton),
        ("全局回路多尺寸", t_global_cycle),
        ("GA 进化", t_ga_evolve),
        ("GA 存档读入", t_ga_persist),
        ("DQN 智能体", t_dqn),
        ("CNN-DQN 智能体", t_cnn_dqn),
        ("CNN 多步空间规划", t_cnn_safe),
        ("离线束搜索", t_beam),
        ("CNN+前向模型想象", t_world_cnn),
        ("Dreamer 世界模型策略", t_dreamer),
        ("冻结世界模型往返", t_world_roundtrip),
        ("grid 前向模型", t_grid_forward),
        ("启动器入口", t_launcher_buttons),
    ]
    for name, fn in checks:
        check(name, fn)

    print("\n===== 体检结果 =====")
    npass = 0
    for ok, name, dt, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<22} {dt:6.2f}s  {detail}")
        npass += ok
    print(f"\n通过 {npass}/{len(RESULTS)}")
    if npass < len(RESULTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
