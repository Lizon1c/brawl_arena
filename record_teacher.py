"""Record teacher-vs-scripted matches to mp4 (same setup as eval_teacher.py:
teacher drives unit 0 deterministically, everyone else runs ScriptedBot).

Usage: record_teacher.py <model.zip> <seed> <out.mp4> [max_seconds]
"""
import sys
import numpy as np
import imageio.v2 as imageio
from stable_baselines3 import PPO

from brawl_arena import BrawlArenaEnv
from brawl_arena.render import render
from train_teacher import DiscreteActionWrapper

model_path = sys.argv[1]
seed = int(sys.argv[2])
out_path = sys.argv[3]
max_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0

model = PPO.load(model_path, device="cpu")
env = DiscreteActionWrapper(BrawlArenaEnv(seed=seed, include_frame=False))
obs, _ = env.reset(seed=seed)
game = env.env.game          # capture AFTER reset: reset() rebuilds the Game
cfg = game.cfg

fps = int(round(1.0 / cfg.dt))
scale = 4
writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)

steps = int(max_seconds / cfg.dt)
info = {}
for i in range(steps):
    frame = render(game, viewer_team=0)
    frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
    writer.append_data(frame)
    a, _ = model.predict(obs, deterministic=True)
    obs, r, terminated, truncated, info = env.step(a)
    if terminated or truncated:
        final = render(game, viewer_team=0)
        final = np.repeat(np.repeat(final, scale, axis=0), scale, axis=1)
        for _ in range(fps):
            writer.append_data(final)
        break

writer.close()
print(f"saved {out_path}, {(i + 1) * cfg.dt:.1f}s, "
      f"winner={info.get('winner')}, gems={info.get('gems')}")
