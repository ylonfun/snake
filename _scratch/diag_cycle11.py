"""诊断 11×11 去角回路。"""
import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
from path_game import hamilton_cycle_corner

W = H = 11
p, ex = hamilton_cycle_corner(W, H)
all_cells = {(x, y) for x in range(W) for y in range(H)} - {ex}
print("长度", len(p), "目标", len(all_cells))
dups = sorted({q for q in p if p.count(q) > 1})
print("重复", dups)
print("缺失", sorted(all_cells - set(p)))
breaks = [(a, b) for a, b in zip(p, p[1:])
          if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1]
print("断点", breaks)
print("首尾相邻?", abs(p[0][0]-p[-1][0]) + abs(p[0][1]-p[-1][1]) == 1, p[0], p[-1])
