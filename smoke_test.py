"""Smoke test: run scripted-bot episodes in every mode, verify env API,
dump one rendered frame per mode to out/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from gymnasium.utils.env_checker import check_env

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import MODES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

state_dims = set()
for mode in MODES:
    env = BrawlArenaEnv(Config(mode=mode), seed=42)
    if mode == "gem_grab":
        check_env(env.unwrapped, skip_render_check=True)
        print("env check: OK  obs:",
              {k: v.shape for k, v in env.observation_space.items()})
    state_dims.add(env.game.state_dim)

    obs, _ = env.reset(seed=42)
    total_r, steps = 0.0, 0
    done = False
    saved_frame = False
    while not done and steps < 5000:
        # controlled unit also driven by the scripted bot for this smoke test
        act = env.teammate_policy.act(env.game, 0)
        gym_act = {"move": act.move.astype(np.float32),
                   "aim": act.aim.astype(np.float32),
                   "shoot": int(act.shoot), "super": int(act.use_super)}
        obs, r, terminated, truncated, info = env.step(gym_act)
        total_r += r
        if steps == 150:
            Image.fromarray(obs["frame"]).save(
                os.path.join(OUT, f"frame_{mode}.png"))
            saved_frame = True
        done = terminated
        steps += 1

    if not saved_frame:  # episode ended before step 150: dump the last frame
        Image.fromarray(obs["frame"]).save(os.path.join(OUT, f"frame_{mode}.png"))
    assert obs["frame"].dtype == np.uint8 and obs["frame"].shape[2] == 3
    assert np.isfinite(obs["state"]).all()
    extra = {k: v for k, v in info.items() if k in ("gems", "scores", "round_wins")}
    print(f"[{mode}] episode: {steps} steps, reward {total_r:.3f}, "
          f"winner={info['winner']}, {extra}, t={info['t']:.1f}s")

assert len(state_dims) == 1, f"state dim differs across modes: {state_dims}"
print(f"state_dim constant across modes: {state_dims.pop()}")
print("smoke test: OK")
