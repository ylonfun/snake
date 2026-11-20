import os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
from train_rl import train

train(total_steps=300000, state_mode="full")
