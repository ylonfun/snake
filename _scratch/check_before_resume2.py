import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
d = np.load(os.path.join("cnn_ckpt", "latest.npz"))
print("latest 含 step:", "step" in d, "值:", int(d["step"]) if "step" in d else None)
h = np.load(os.path.join("cnn_ckpt", "history.npz"))
print("history 点数:", len(h["step"]), "最后步:", int(h["step"][-1]))
print("best 是否已存在:", os.path.exists(os.path.join("cnn_ckpt", "best.npz")))
