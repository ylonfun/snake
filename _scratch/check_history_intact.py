import os, sys
root = r"D:\Duobao-workpalce\NN-net-learn\snake-main"
os.chdir(root); sys.path.insert(0, root)
import numpy as np
h = np.load(os.path.join("cnn_ckpt", "history.npz"))
print("history step:", list(h["step"]))
print("history eval_score:", [round(x,1) for x in h["eval_score"]])
d = np.load(os.path.join("cnn_ckpt", "latest.npz"))
print("latest 是否含 step 字段:", "step" in d, "文件数组键数量:", len(d.files))
