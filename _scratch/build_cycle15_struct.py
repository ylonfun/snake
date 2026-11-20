"""15×15 去角 (0,0) 结构化 Hamilton 回路构造（双列锯齿闭合）+ 严格验证。"""
W = H = 15
EXCLUDE = (0, 0)
all_cells = {(x, y) for x in range(W) for y in range(H)} - {EXCLUDE}


def build():
    p = [(0, 1), (1, 1), (1, 0)]                       # 左上角局部
    p += [(x, 0) for x in range(2, W)]                 # 第一行向右
    p += [(W - 1, y) for y in range(1, H)]             # 最后一列向下
    # 蛇形列 W-2 .. 2
    for k, c in enumerate(range(W - 2, 1, -1)):
        if k % 2 == 0:                                  # 列13: 向上
            p += [(c, y) for y in range(H - 1, 0, -1)]
        else:                                           # 列12: 向下
            p += [(c, y) for y in range(1, H)]
    # 此时末格为 (2,14)；最左双列锯齿（覆盖列1 与第一列各 13 格）
    assert p[-1] == (2, 14), p[-1]
    p += [(1, 14), (0, 14)]
    for y in range(13, 1, -1):
        if y % 2 == 1:                              # 奇数行：先0后1
            p += [(0, y), (1, y)]
        else:                                       # 偶数行：先1后0
            p += [(1, y), (0, y)]
    # 终点 (0,2)，与 (0,1) 闭合
    assert p[-1] == (0, 2), p[-1]
    return p


p = build()
assert len(p) == len(all_cells), (len(p), len(all_cells))
assert len(set(p)) == len(all_cells), "每格恰好一次"
assert not (all_cells - set(p)), "无缺失"
for a, b in zip(p, p[1:] + p[:1]):
    assert abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1, (a, b)
print("严格验证通过：去角哈密顿回路长度 =", len(p))
import json
with open(r"D:\Duobao-workpalce\NN-net-learn\snake-main\_scratch\cycle15.json", "w") as f:
    json.dump(p, f)
print("已保存 cycle15.json，首", p[0], "尾", p[-1])
