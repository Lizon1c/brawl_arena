"""Numeric replay of teacher vs scripted bots: log game state every 15s.

Per sample: sim time, gems per team, alive per team, unit0 (teacher) alive,
unit0 gems held, unit0 distance to mine. Plus per-game death count of unit0.
"""
import numpy as np
from stable_baselines3 import PPO

from brawl_arena import BrawlArenaEnv
from train_teacher import DiscreteActionWrapper

model = PPO.load("runs/teacher_10m_v2/selfplay_latest.zip", device="cpu")

for seed in range(5000, 5003):
    env = DiscreteActionWrapper(BrawlArenaEnv(seed=seed, include_frame=False))
    obs, _ = env.reset(seed=seed)
    g = env.env.game
    mine = np.array([g.cfg.map_w / 2.0, g.cfg.map_h / 2.0])
    deaths0 = 0
    was_alive = True
    step = 0
    done = False
    print(f"--- seed {seed} ---")
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        step += 1
        u0 = g.units[0]
        if was_alive and not u0.alive:
            deaths0 += 1
        was_alive = u0.alive
        if step % 225 == 0 or done:   # every 15 s
            alive = [sum(1 for u in g.units if u.team == t and u.alive)
                     for t in (0, 1)]
            held = getattr(u0, "gems", 0)
            d = np.linalg.norm(u0.pos - mine)
            print(f"  t={g.t:5.1f}s gems={info.get('gems')} alive={alive} "
                  f"u0_alive={u0.alive} u0_gems={held} u0_dist_mine={d:.1f}")
    print(f"  result: winner={info['winner']} u0_deaths={deaths0}")
