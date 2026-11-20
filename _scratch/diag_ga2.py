import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
from compare import rollout_ga
genes = np.array(list(map(float, open(os.path.join("genes", "record", "genes")).read().split())))
res = [rollout_ga(genes, s) for s in range(30)]
scores = sorted([r[0] for r in res], reverse=True)
print("GA 30局分数：", scores)
print("超过50分的局数：", sum(1 for s in scores if s > 50))
