import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
os.environ["SDL_VIDEODRIVER"] = "dummy"
from compare import rollout_ham
for seed in range(3):
    print("哈密顿 seed", seed, "（分数,步数）=", rollout_ham(seed))
