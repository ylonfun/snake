"""train_rl.train 真实循环冒烟：在临时目录里跑，避免覆盖 dqn_ckpt 生产存档。"""
import os
import sys
import tempfile

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import train_rl

work = tempfile.mkdtemp(prefix="train_rl_smoke_")
os.chdir(work)                      # ckpt_dir 是相对路径，文件全部落在这里
print("临时工作目录:", work)

agent = train_rl.train(total_steps=260, interval=130, state_mode="local")
print(f"[PASS] train_rl 真实循环跑通：步数={agent.step_count} "
      f"buffer={agent.buffer.size} eps={agent.eps:.3f}")
print("存档文件:", sorted(os.listdir(os.path.join(work, "dqn_ckpt"))) if
      os.path.exists(os.path.join(work, "dqn_ckpt")) else "无")
print("OK")
