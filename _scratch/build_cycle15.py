"""15×15 去掉角格 (0,0) 后，在 224 格上用 Warnsdorff 搜索哈密顿回路。"""
import random, time

W = H = 15
cells = [(x, y) for x in range(W) for y in range(H) if (x, y) != (0, 0)]
cell_set = set(cells)
N = len(cells)
assert N == 224


def nbrs(p):
    x, y = p
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = (x + dx, y + dy)
        if q in cell_set:
            out.append(q)
    return out


def search(seed, deadline_s=60):
    rnd = random.Random(seed)
    t0 = time.time()
    start = (1, 0)
    path = [start]
    used = {start}
    while len(path) < N:
        if time.time() - t0 > deadline_s:
            return None
        cur = path[-1]
        cand = [q for q in nbrs(cur) if q not in used]
        if not cand:
            return None
        # Warnsdorff：后续度最小优先；度数相同则随机打破平局
        def degree(q):
            return sum(1 for z in nbrs(q) if z not in used)
        cand.sort(key=lambda q: (degree(q), rnd.random()))
        path.append(cand[0]); used.add(cand[0])
    # 闭合：末格须与起点相邻
    if start in nbrs(path[-1]):
        return path
    return None


for seed in range(200):
    r = search(seed)
    if r is not None:
        # 严格验证
        assert len(r) == N and len(set(r)) == N
        for a, b in zip(r, r[1:] + r[:1]):
            assert abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
        print("找到合法去角哈密顿回路！seed=", seed, "长度=", len(r))
        print("起点", r[0], "终点", r[-1])
        import json
        with open(r"D:\Duobao-workpalce\NN-net-learn\snake-main\_scratch\cycle15.json", "w") as f:
            json.dump(r, f)
        break
else:
    print("200 次尝试未找到")
