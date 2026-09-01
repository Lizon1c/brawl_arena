"""Record a scripted-bot match to mp4 at normal speed (15 fps, upscaled x4)."""
import sys
import numpy as np
import imageio.v2 as imageio

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.render import render

mode = sys.argv[1] if len(sys.argv) > 1 else "gem_grab"
out_path = sys.argv[2] if len(sys.argv) > 2 else f"out/match_{mode}.mp4"
max_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
map_name = sys.argv[4] if len(sys.argv) > 4 else None

if map_name:
    # force a specific custom map: pin sample_map to this entry
    import brawl_arena.maps as maps_mod
    entry = next(e for e in maps_mod.CUSTOM_MAPS.get(mode, [])
                 if e["name"] == map_name)
    tiles0, spawns0 = entry["tiles"], entry["spawns"]
    maps_mod.sample_map = lambda mode_, rng: (tiles0.copy(), spawns0)

cfg = Config(mode=mode)
env = BrawlArenaEnv(config=cfg, seed=2024)
env.reset(seed=2024)

fps = int(round(1.0 / cfg.dt))
scale = 4
writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)

steps = int(max_seconds / cfg.dt)
info = {}
for i in range(steps):
    frame = render(env.game, viewer_team=0)
    frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
    writer.append_data(frame)
    act = env.teammate_policy.act(env.game, 0)
    gym_act = {"move": act.move.astype(np.float32), "aim": act.aim.astype(np.float32),
               "shoot": int(act.shoot), "super": int(act.use_super)}
    obs, r, terminated, truncated, info = env.step(gym_act)
    if terminated:
        # append the final state for a second so the result is visible
        final = render(env.game, viewer_team=0)
        final = np.repeat(np.repeat(final, scale, axis=0), scale, axis=1)
        for _ in range(fps):
            writer.append_data(final)
        break

writer.close()
print(f"saved {out_path}, {i + 1} sim steps ({(i + 1) * cfg.dt:.1f}s), "
      f"winner={info.get('winner')}, info={ {k: v for k, v in info.items() if k != 't'} }")
