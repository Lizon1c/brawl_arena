"""Diagnose idle bot: track per-unit movement on the recorded seed."""
import numpy as np
from brawl_arena import BrawlArenaEnv, Config, ScriptedBot

cfg = Config(mode="gem_grab")
env = BrawlArenaEnv(config=cfg, seed=2024)
env.reset(seed=2024)
bot = ScriptedBot()

last = [u.pos.copy() for u in env.game.units]
idle = np.zeros(6)
for step in range(1800):
    acts = [bot.act(env.game, i) for i in range(6)]
    gym_act = {"move": acts[0].move.astype(np.float32),
               "aim": acts[0].aim.astype(np.float32),
               "shoot": int(acts[0].shoot), "super": int(acts[0].use_super)}
    from brawl_arena.core import Action
    env.game.step(acts)
    for i, u in enumerate(env.game.units):
        if u.alive and np.linalg.norm(u.pos - last[i]) < 0.02:
            idle[i] += 1
        last[i] = u.pos.copy()

for i, u in enumerate(env.game.units):
    tile = env.game.tiles[int(u.pos[1]), int(u.pos[0])]
    print(f"unit{i} team{u.team} {u.archetype:8s} idle_steps={int(idle[i]):4d} "
          f"pos=({u.pos[0]:.1f},{u.pos[1]:.1f}) spawn=({u.spawn[0]:.1f},{u.spawn[1]:.1f}) "
          f"gems={u.gems} tile={tile}")
