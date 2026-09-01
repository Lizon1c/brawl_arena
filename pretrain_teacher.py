"""Pretrain the teacher by behaviour-cloning the scripted bot.

Collects (privileged state, scripted-bot action) pairs with the scripted bot
driving unit 0 (everyone else is also scripted), then trains the PPO
policy's action heads by cross-entropy. The resulting model is a warm start
for self-play: `train_teacher.py --init runs/teacher_bc/bc_init.zip`.

Usage:
    .venv/Scripts/python pretrain_teacher.py --samples 100000 --epochs 10
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import MODES
from train_teacher import (DiscreteActionWrapper, StateOnlyExtractor, _DIRS8)

ACTION_DIMS = (9, 9, 2, 2)


def action_to_multi(move: np.ndarray, aim: np.ndarray, shoot: bool,
                    use_super: bool) -> np.ndarray:
    """Inverse of multi_to_action: quantize to the 9/9/2/2 indices."""
    mv_idx = 8
    if np.linalg.norm(move) > 0.5:
        mv_idx = int(np.argmax(_DIRS8 @ np.asarray(move, dtype=np.float32)))
    aim_idx = 8
    if np.linalg.norm(aim) > 0.5:
        aim_idx = int(np.argmax(_DIRS8 @ np.asarray(aim, dtype=np.float32)))
    return np.array([mv_idx, aim_idx, int(shoot), int(use_super)],
                    dtype=np.int64)


def collect(mode: str, n: int, seed: int):
    env = BrawlArenaEnv(config=Config(mode=mode), seed=seed,
                        include_frame=False)
    states = np.zeros((n, env.observation_space["state"].shape[0]),
                      dtype=np.float32)
    acts = np.zeros((n, 4), dtype=np.int64)
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    game = env.game   # reset() rebuilds the Game; re-grab on every reset
    for i in range(n):
        bot_act = env.teammate_policy.act(game, 0)
        states[i] = game.state_vector(0)
        a = action_to_multi(bot_act.move, bot_act.aim, bot_act.shoot,
                            bot_act.use_super)
        acts[i] = a
        obs, _, done, _, _ = env.step(
            {"move": bot_act.move.astype(np.float32),
             "aim": bot_act.aim.astype(np.float32),
             "shoot": int(bot_act.shoot), "super": int(bot_act.use_super)})
        if done:
            obs, _ = env.reset()
            game = env.game
        if (i + 1) % 20000 == 0:
            print(f"  collected {i + 1}/{n} "
                  f"({(i + 1) / (time.perf_counter() - t0):.0f}/s)")
    return states, acts


def make_model(mode: str, device: str) -> PPO:
    """Identical architecture to train_teacher.main."""
    def _init():
        env = BrawlArenaEnv(config=Config(mode=mode), seed=0,
                            include_frame=False)
        return DiscreteActionWrapper(env)
    env = DummyVecEnv([_init])
    return PPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=StateOnlyExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        ),
        learning_rate=3e-4, n_steps=1024, batch_size=512,
        gamma=0.995, verbose=0, device=device,
    )


def train_bc(model: PPO, states: np.ndarray, acts: np.ndarray, epochs: int,
             device: str, batch_size: int = 1024, lr: float = 1e-3):
    policy = model.policy
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    x_all = torch.from_numpy(states)
    y_all = torch.from_numpy(acts)
    n = len(states)
    n_val = max(1, n // 10)
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    for ep in range(epochs):
        order = np.random.permutation(len(tr_idx))
        tot = nb = 0
        for b in range(0, len(order), batch_size):
            idx = tr_idx[order[b:b + batch_size]]
            x = x_all[idx].to(device)
            y = y_all[idx].to(device)
            obs = {"state": x}
            feats = policy.extract_features(obs)
            lat_pi, _ = policy.mlp_extractor(feats)
            logits = policy.action_net(lat_pi)
            heads = torch.split(logits, list(ACTION_DIMS), dim=1)
            loss = sum(torch.nn.functional.cross_entropy(h, y[:, k])
                       for k, h in enumerate(heads))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        # validation accuracy per head
        policy.eval()
        correct = np.zeros(4)
        with torch.no_grad():
            for b in range(0, len(val_idx), 4096):
                idx = val_idx[b:b + 4096]
                x = x_all[idx].to(device)
                feats = policy.extract_features({"state": x})
                lat_pi, _ = policy.mlp_extractor(feats)
                logits = policy.action_net(lat_pi)
                heads = torch.split(logits, list(ACTION_DIMS), dim=1)
                for k, h in enumerate(heads):
                    correct[k] += (h.argmax(1).cpu() == y_all[idx, k]).sum()
        policy.train()
        acc = correct / len(val_idx)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / nb:.4f} "
              f"val acc move {acc[0]:.3f} aim {acc[1]:.3f} "
              f"shoot {acc[2]:.3f} super {acc[3]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="gem_grab", choices=MODES)
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="runs/teacher_bc")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[collect] {args.samples} samples from scripted bot...")
    states, acts = collect(args.mode, args.samples, seed=42)
    print(f"[model] building PPO policy on {args.device}...")
    model = make_model(args.mode, args.device)
    print(f"[bc] training {args.epochs} epochs...")
    train_bc(model, states, acts, args.epochs, args.device)
    out = os.path.join(args.out, "bc_init")
    model.save(out)
    print("saved to", out + ".zip")


if __name__ == "__main__":
    main()
