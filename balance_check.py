"""Per-map balance analysis: Bayesian win rates from bot-vs-bot simulation.

For every custom map (and optionally built-ins), simulates N bot matches and
reports, with a Beta(1,1) posterior:
  - side win rate (team 0 vs team 1) -- catches asymmetric maps
  - per-archetype win rate -- catches "this hero is broken on this map"
  - average winning margin -- catches one-sided stomps

Usage: .venv/Scripts/python balance_check.py [games_per_map]
"""
import math
import multiprocessing as mp
import os
import sys

import numpy as np

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
STEP_CAP = 4000
SHOWDOWN_UNITS = 6


def _run_map(args):
    mode, name, tiles, spawns, n_games, seed0 = args
    import brawl_arena.maps as maps
    maps.sample_map = lambda mode_, rng: (tiles.copy(), spawns)
    from brawl_arena.core import Game, Config
    from brawl_arena.bots import ScriptedBot

    bot = ScriptedBot(seed=seed0)
    side_w = side_l = 0
    margins = []
    arch = {}  # archetype -> [wins, games]
    for g in range(n_games):
        game = Game(Config(mode=mode), seed=seed0 + g)
        while not game.done and game.t / game.cfg.dt < STEP_CAP:
            game.step(bot.act_batch(game, list(range(len(game.units)))))
        w = game.winner
        if mode == "showdown":
            if w is not None:
                wa = game.units[w].archetype
                for u in game.units:
                    a = arch.setdefault(u.archetype, [0, 0])
                    a[1] += 1
                    if u.archetype == wa:
                        a[0] += 1
        else:
            if w is not None:
                if w == 0:
                    side_w += 1
                else:
                    side_l += 1
            for u in game.units:
                a = arch.setdefault(u.archetype, [0, 0])
                a[1] += 1
                if w is not None and u.team == w:
                    a[0] += 1
            if mode == "gem_grab":
                held = [sum(u.gems for u in game.units if u.team == t) for t in (0, 1)]
                margins.append(abs(held[0] - held[1]))
            elif mode == "brawl_ball":
                margins.append(abs(game.scores[0] - game.scores[1]))
            elif mode == "knockout":
                margins.append(abs(game.round_wins[0] - game.round_wins[1]))
    return mode, name, side_w, side_l, n_games, margins, arch


def _beta(wins, losses):
    a, b = wins + 1.0, losses + 1.0
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1.0))
    sd = math.sqrt(var)
    return mean, max(0.0, mean - 1.96 * sd), min(1.0, mean + 1.96 * sd)


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from brawl_arena.maps import CUSTOM_MAPS

    jobs = []
    for mode, entries in CUSTOM_MAPS.items():
        for e in entries:
            jobs.append((mode, e["name"], e["tiles"], e["spawns"], GAMES, 777))

    with mp.Pool(min(8, len(jobs))) as pool:
        results = pool.map(_run_map, jobs)

    for mode, name, sw, sl, n, margins, arch in results:
        print(f"\n== {name} ({n} games) ==")
        if mode != "showdown":
            m_, lo, hi = _beta(sw, sl)
            flag = "  <-- SIDE BIAS" if lo > 0.5 or hi < 0.5 else ""
            print(f"  team0 win rate: {m_:.3f}  95%CI [{lo:.3f}, {hi:.3f}]{flag}")
            if margins:
                print(f"  avg margin: {np.mean(margins):.2f}  max: {max(margins)}")
        expected = 1.0 / SHOWDOWN_UNITS if mode == "showdown" else 0.5
        for a_name, (wins, games) in sorted(arch.items()):
            m_, lo, hi = _beta(wins, games - wins)
            if mode == "showdown":
                flag = "  <-- HERO BIAS" if lo > expected * 1.5 or hi < expected / 1.5 else ""
                print(f"  {a_name:8s}: win share {m_:.3f} (fair={expected:.3f}) "
                      f"95%CI [{lo:.3f}, {hi:.3f}]{flag}")
            else:
                flag = "  <-- HERO BIAS" if lo > 0.5 or hi < 0.5 else ""
                print(f"  {a_name:8s}: win rate {m_:.3f}  95%CI [{lo:.3f}, {hi:.3f}]{flag}")


if __name__ == "__main__":
    main()
