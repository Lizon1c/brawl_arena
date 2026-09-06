"""Tests for mid-game snapshot + endgame-start (see student_rl snapshots).

1. deepcopy / pickle consistency: branch a mid-game Game, run both copies
   with identical actions, require identical trajectories (pos, hp, gems,
   ball, t, rng continuation).
2. End-to-end env restore: collect snapshots via student_rl._snapshot_trigger,
   restore with BrawlArenaEnv.set_game(deepcopy), keep playing; check obs,
   reward baselines (no first-step spike) and the id(game) change that
   resets FrozenStudentPolicy's feature cache.

Run: .venv/Scripts/python test_snapshot.py
"""
import copy
import pickle

import numpy as np

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.bots import ScriptedBot
from brawl_arena.core import MODES, Action, Game
from student_rl import SNAPSHOT_CAP, _snapshot_trigger


def rand_actions(rng, n):
    acts = []
    for _ in range(n):
        acts.append(Action(
            move=rng.uniform(-1, 1, 2), aim=rng.uniform(-1, 1, 2),
            shoot=bool(rng.random() < 0.5), use_super=bool(rng.random() < 0.1)))
    return acts


def signature(g: Game):
    return {
        "t": g.t, "done": g.done, "winner": g.winner,
        "pos": np.array([u.pos for u in g.units]),
        "hp": np.array([u.hp for u in g.units]),
        "ammo": np.array([u.ammo for u in g.units]),
        "alive": [u.alive for u in g.units],
        "gems_held": [u.gems for u in g.units],
        "gems": sorted(tuple(np.round(x, 6)) for x in g.gems),
        "n_proj": len(g.projectiles),
        "proj": sorted((p.owner, round(p.damage, 3),
                        round(p.pos[0], 6), round(p.pos[1], 6))
                       for p in g.projectiles),
        "ball_pos": g.ball_pos.copy(), "ball_carrier": g.ball_carrier,
        "scores": list(g.scores), "round_wins": list(g.round_wins),
        "countdown": (g.countdown_team, round(g.countdown_t, 6)),
        "poison": g.poison_radius,
        "tiles": g.tiles.copy(),
    }


def sig_equal(a, b):
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray):
            if not np.array_equal(va, vb):
                return f"field {k} differs"
        elif va != vb:
            return f"field {k} differs: {va!r} vs {vb!r}"
    return None


def test_consistency():
    for mode in MODES:
        g = Game(Config(mode=mode), seed=42)
        rng = np.random.default_rng(999)
        for _ in range(250):          # reach mid-game
            g.step(rand_actions(rng, len(g.units)))
        assert not g.done or mode == "knockout"  # done is fine too, just info

        # --- deepcopy branch ---
        fork = copy.deepcopy(g)
        plan = [rand_actions(rng, len(g.units)) for _ in range(150)]
        for branch in (g, fork):
            for acts in plan:
                branch.step(acts)
        err = sig_equal(signature(g), signature(fork))
        assert err is None, f"[{mode}] deepcopy divergence: {err}"
        # rng state must have been copied too (spread noise, ball ties...)
        assert g.rng.random() == fork.rng.random(), f"[{mode}] rng diverged"

        # --- pickle round-trip on a fresh mid-game state ---
        g2 = Game(Config(mode=mode), seed=42)
        rng2 = np.random.default_rng(999)
        for _ in range(250):
            g2.step(rand_actions(rng2, len(g2.units)))
        blob = pickle.dumps(g2)
        g3 = pickle.loads(blob)
        plan2 = [rand_actions(rng2, len(g2.units)) for _ in range(150)]
        for branch in (g2, g3):
            for acts in plan2:
                branch.step(acts)
        err = sig_equal(signature(g2), signature(g3))
        assert err is None, f"[{mode}] pickle divergence: {err}"
        print(f"[{mode}] deepcopy+pickle OK, snapshot bytes={len(blob)}")


def test_env_restore():
    for mode in MODES:
        env = BrawlArenaEnv(config=Config(mode=mode, reward_shaping=True),
                            seed=7, include_frame=False)
        bot = ScriptedBot(seed=3)     # drives unit 0
        obs, _ = env.reset()
        snaps = {}                    # trigger class -> deepcopied game
        for _ in range(4000):
            a = bot.act(env.game, 0)
            obs, r, done, _, info = env.step({
                "move": a.move.astype(np.float32),
                "aim": a.aim.astype(np.float32),
                "shoot": int(a.shoot), "super": int(a.use_super)})
            if done:
                env.reset()
                continue
            trig = _snapshot_trigger(env.game)
            if trig and trig not in snaps:
                snaps[trig] = copy.deepcopy(env.game)
        assert snaps, f"[{mode}] no snapshot triggered in 4000 steps"

        for trig, snap in snaps.items():
            assert not snap.done
            assert len(pickle.dumps(snap)) < 100_000
            old_game = env.game
            obs, _ = env.set_game(copy.deepcopy(snap))
            # id(game) changed -> FrozenStudentPolicy cache resets itself
            assert id(env.game) != id(old_game)
            assert env.game is not snap   # fresh copy, bucket entry untouched
            assert id(env.game) != id(snap)
            assert obs["state"].shape == (env.game.state_dim,)
            assert np.isfinite(obs["state"]).all()
            # differential-reward baselines seeded from restored state
            if mode == "gem_grab":
                mine, enemy = env.game._relative_score(0)
                assert env._last_mine == mine and env._last_enemy == enemy
            rs = []
            for _ in range(100):
                a = bot.act(env.game, 0)
                obs, r, done, _, _ = env.step({
                    "move": a.move.astype(np.float32),
                    "aim": a.aim.astype(np.float32),
                    "shoot": int(a.shoot), "super": int(a.use_super)})
                rs.append(r)
                assert np.isfinite(obs["state"]).all()
                if done:
                    break
            # no reward spike on the first step after restore (baselines ok)
            assert abs(rs[0]) < 0.5, f"[{mode}/{trig}] first r={rs[0]}"
            print(f"[{mode}/{trig}] restore OK: t={snap.t:.1f}s, "
                  f"first r={rs[0]:+.4f}, 100-step |r|max={max(map(abs, rs)):.3f}")


if __name__ == "__main__":
    print(f"SNAPSHOT_CAP={SNAPSHOT_CAP}")
    test_consistency()
    test_env_restore()
    print("ALL SNAPSHOT TESTS PASSED")
