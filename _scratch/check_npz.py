import numpy as np
new = np.load(r"D:\Duobao-workpalce\NN-net-learn\snake-main\world_frozen.npz")
old = np.load(r"D:\Duobao-workpalce\NN-net-learn\snake-main\_backup_global_v1\world_frozen.npz")
for k in ["R_W0", "R_W1", "R_b1"]:
    same = np.allclose(new[k], old[k])
    print(k, "新旧相同" if same else "新旧不同", "新均值", round(float(new[k].mean()), 5))
