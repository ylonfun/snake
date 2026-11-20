import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
from train_rl import train, CKPT_DIR

train(total_steps=1000000, resume=os.path.join(CKPT_DIR, "latest.npz"))
