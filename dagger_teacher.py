"""DAgger: iterative imitation of the scripted bot with the LEARNER driving.

BC's failure mode is covariate shift: the bot never visits the states a
slightly-wrong learner ends up in. DAgger fixes this by letting the learner
drive (mixed with the oracle early on, beta decays to 0), while every
visited state is labelled by the scripted bot. Data is aggregated across
rounds and the policy retrained supervised -- no value net, no advantage
estimates, no self-play bubble.

Each round saves a PPO zip (same architecture as train_teacher.py), directly
evaluable with:  eval_teacher.py runs/teacher_dagger/round_N.zip 30

Usage:
    .venv/Scripts/python dagger_teacher.py --rounds 8 --per-round 20000
"""
import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import MODES
from pretrain_teacher import action_to_multi, make_model, train_bc
from train_teacher import multi_to_action


def collect_dagger(model: PPO, mode: str, n: int, seed: int, beta: float):
    """Learner drives unit 0 (oracle takes over with prob beta); every
    visited state is labelled by the scripted bot oracle."""
    env = BrawlArenaEnv(config=Config(mode=mode), seed=seed,
                        include_frame=False)
    obs, _ = env.reset(seed=seed)
    game = env.game
    states = np.zeros((n, env.observation_space["state"].shape[0]),
                      dtype=np.float32)
    acts = np.zeros((n, 4), dtype=np.int64)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for i in range(n):
        oracle = env.teammate_policy.act(game, 0)
        states[i] = game.state_vector(0)
        acts[i] = action_to_multi(oracle.move, oracle.aim, oracle.shoot,
                                  oracle.use_super)
        if rng.random() < beta:
            gym_act = {"move": oracle.move.astype(np.float32),
                       "aim": oracle.aim.astype(np.float32),
                       "shoot": int(oracle.shoot),
                       "super": int(oracle.use_super)}
        else:
            t_obs = {"frame": np.zeros((1, 1, 3), dtype=np.uint8),
                     "state": game.state_vector(0)[None]}
            a, _ = model.predict(t_obs, deterministic=False)
            act = multi_to_action(np.asarray(a).ravel())
            gym_act = {"move": act.move, "aim": act.aim,
                       "shoot": int(act.shoot), "super": int(act.use_super)}
        obs, _, done, _, _ = env.step(gym_act)
        if done:
            obs, _ = env.reset()
            game = env.game
        if (i + 1) % 10000 == 0:
            print(f"    collected {i + 1}/{n} "
                  f"({(i + 1) / (time.perf_counter() - t0):.0f}/s)")
    return states, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="gem_grab", choices=MODES)
    ap.add_argument("--init", type=str, default="runs/teacher_bc/bc_init.zip")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--per-round", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=5,
                    help="supervised epochs per round on the aggregate set")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="runs/teacher_dagger")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = make_model(args.mode, args.device)
    model.set_parameters(args.init)
    print(f"init from {args.init}")

    all_states, all_acts = [], []
    for r in range(args.rounds):
        # oracle mixing decays linearly to 0 over the first half of rounds
        beta = max(0.0, 0.5 - 0.5 * r / max(1, args.rounds // 2))
        print(f"[round {r}] collecting {args.per_round} samples "
              f"(beta={beta:.2f})...")
        s, a = collect_dagger(model, args.mode, args.per_round,
                              seed=100 + r, beta=beta)
        all_states.append(s)
        all_acts.append(a)
        states = np.concatenate(all_states)
        acts = np.concatenate(all_acts)
        print(f"[round {r}] retraining on {len(states)} samples...")
        train_bc(model, states, acts, args.epochs, args.device)
        path = os.path.join(args.out, f"round_{r}")
        model.save(path)
        print(f"[round {r}] saved {path}.zip")


if __name__ == "__main__":
    main()
