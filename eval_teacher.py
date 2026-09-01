"""Evaluate a trained teacher against the scripted bot."""
import sys
import numpy as np
from stable_baselines3 import PPO

from brawl_arena import BrawlArenaEnv
from train_teacher import DiscreteActionWrapper

MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/teacher_gem_grab/teacher_final.zip"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

model = PPO.load(MODEL, device="cuda")
wins = draws = losses = 0
gem_diffs = []
for seed in range(N):
    env = DiscreteActionWrapper(BrawlArenaEnv(seed=5000 + seed, include_frame=False))
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
    w = info["winner"]
    if w == 0:
        wins += 1
    elif w == 1:
        losses += 1
    else:
        draws += 1
    gem_diffs.append(info["gems"][0] - info["gems"][1])
print(f"{MODEL}: vs scripted bot over {N} games: W{wins} D{draws} L{losses}, "
      f"avg gem diff {np.mean(gem_diffs):.1f}")
