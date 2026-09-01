"""Omnidirectional student: continuous move/aim joysticks instead of 9-way.

Same trunk as student_vision.py (DINOv2-S/14 fully fine-tuned -> per-frame
384-d cls token -> OscillatorBank CPG-style motion encoding -> readout),
but the action heads match a real gamepad:
  move  : Linear(256, 2) + tanh      (vector in the unit disk)
  aim   : Linear(256, 2)             (unit direction, MSE on unit vectors)
  shoot : Linear(256, 2) CE          (button)
  super : Linear(256, 2) CE          (button)

Data: frames are REUSED from runs/student_vision/bc_data.npz (the scripted
bot and env are fully seeded, so re-running collection with the same seeds
reproduces the same trajectory). --collect-actions only re-derives the
continuous labels and verifies alignment against the discrete ones.

Usage:
    .venv/Scripts/python student_omni.py --collect-actions 40000
    .venv/Scripts/python student_omni.py --train 40 --init runs/student_vision/dino_cpg_ep20.pt
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import MODES
from pretrain_teacher import action_to_multi
from student_vision import (DATA, IMG, DINO_DIM, OscillatorBank, make_windows)

OUT = os.path.join("runs", "student_vision")
ACTS_OMNI = os.path.join(OUT, "bc_acts_omni.npz")


# ------------------------------------------------------------------ model
class OmniStudent(nn.Module):
    """DINOv2-S/14 + oscillator bank + gamepad heads (continuous sticks)."""

    def __init__(self, n_modes: int = 128, n_frames: int = 16):
        super().__init__()
        import torch.hub
        self.backbone = torch.hub.load("facebookresearch/dinov2",
                                       "dinov2_vits14", pretrained=True)
        self.n_frames = n_frames
        self.osc = OscillatorBank(DINO_DIM, n_modes)
        self.readout = nn.Sequential(nn.Linear(2 * n_modes, 256), nn.ReLU())
        self.move_head = nn.Linear(256, 2)
        self.aim_head = nn.Linear(256, 2)
        self.shoot_head = nn.Linear(256, 2)
        self.super_head = nn.Linear(256, 2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) floats [0,255] -> (B, 256) readout."""
        B, T = x.shape[:2]
        v = x.reshape(B * T, *x.shape[2:]) / 255.0
        v = (v - self.mean) / self.std
        v = F.interpolate(v, size=(IMG, IMG), mode="bilinear",
                          align_corners=False)
        feats = self.backbone.forward_features(v)["x_norm_clstoken"]
        feats = feats.reshape(B, T, DINO_DIM)
        return self.readout(self.osc(feats))

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        return {
            "move": torch.tanh(self.move_head(z)),
            "aim": self.aim_head(z),
            "shoot": self.shoot_head(z),
            "super": self.super_head(z),
        }

    def load_trunk(self, path: str):
        """Warm start backbone/osc/readout from a discrete-head checkpoint."""
        sd = torch.load(path, map_location="cpu")
        own = self.state_dict()
        ok = {k: v for k, v in sd.items()
              if k in own and own[k].shape == v.shape}
        own.update(ok)
        self.load_state_dict(own)
        print(f"warm start: loaded {len(ok)}/{len(sd)} tensors from {path}")


# ------------------------------------------------------- action collection
def collect_actions(n: int, seed: int = 42):
    """Re-run the exact collection trajectory of student_vision.collect but
    store the continuous bot vectors. Verification: the re-discretized labels
    and episode boundaries must match bc_data.npz exactly."""
    moves = np.zeros((n, 2), dtype=np.float32)
    aims = np.zeros((n, 2), dtype=np.float32)
    btns = np.zeros((n, 2), dtype=np.int64)
    ep_start = np.zeros(n, dtype=bool)
    i = 0
    t0 = time.perf_counter()
    for round_ in range(9999):
        for mode in MODES:
            if i >= n:
                break
            env = BrawlArenaEnv(config=Config(mode=mode),
                                seed=seed + round_, include_frame=False)
            env.reset(seed=seed + round_)
            game = env.game
            ep_start[i] = True
            done = False
            while not done and i < n:
                bot = env.teammate_policy.act(game, 0)
                moves[i] = bot.move
                a = np.asarray(bot.aim, dtype=np.float32)
                aims[i] = a / (np.linalg.norm(a) + 1e-6)
                btns[i] = (int(bot.shoot), int(bot.use_super))
                _, _, done, _, _ = env.step(
                    {"move": bot.move.astype(np.float32),
                     "aim": bot.aim.astype(np.float32),
                     "shoot": int(bot.shoot), "super": int(bot.use_super)})
                i += 1
                if i % 10000 == 0:
                    print(f"  {i}/{n} ({i / (time.perf_counter() - t0):.0f}/s)")
        if i >= n:
            break
    # alignment check against the discrete dataset (frames were rendered
    # from the same seeded trajectory, so labels must match exactly)
    d = np.load(DATA)
    ref = d["acts"]
    redisc = np.stack([action_to_multi(moves[k], aims[k], btns[k, 0],
                                       btns[k, 1]) for k in range(n)])
    n_mm = int((redisc != ref).any(axis=1).sum())
    ep_mm = int((ep_start != d["ep_start"]).sum())
    print(f"alignment: {n_mm}/{n} label mismatches, {ep_mm} ep_start diffs")
    if n_mm > n // 1000 or ep_mm > 0:
        raise SystemExit("trajectory drifted from the rendered dataset; "
                         "re-collect frames instead of reusing them")
    np.savez_compressed(ACTS_OMNI, moves=moves, aims=aims, btns=btns)
    print(f"saved {ACTS_OMNI}")


# ------------------------------------------------- frame + label collection
def collect_frames(n: int, seed_base: int, out_path: str):
    """Collect a fresh dataset shard (frames + continuous labels) with a new
    seed base so episodes differ from the original bc_data.npz (seed 42).
    Same rendering pipeline as student_vision.collect (90x126, letterboxed)."""
    frames = np.zeros((n, 90, 126, 3), dtype=np.uint8)
    moves = np.zeros((n, 2), dtype=np.float32)
    aims = np.zeros((n, 2), dtype=np.float32)
    btns = np.zeros((n, 2), dtype=np.int64)
    ep_start = np.zeros(n, dtype=bool)
    i = 0
    t0 = time.perf_counter()
    for round_ in range(9999):
        for mode in MODES:
            if i >= n:
                break
            env = BrawlArenaEnv(config=Config(mode=mode),
                                seed=seed_base + round_, include_frame=True)
            obs, _ = env.reset(seed=seed_base + round_)
            game = env.game
            ep_start[i] = True
            done = False
            while not done and i < n:
                bot = env.teammate_policy.act(game, 0)
                f = obs["frame"]
                if f.shape[0] != 90 or f.shape[1] != 126:
                    ff = torch.from_numpy(f).permute(2, 0, 1)[None].float()
                    ff = F.interpolate(ff, size=(90, 126), mode="bilinear",
                                       align_corners=False)
                    f = ff[0].permute(1, 2, 0).byte().numpy()
                frames[i] = f
                moves[i] = bot.move
                a = np.asarray(bot.aim, dtype=np.float32)
                aims[i] = a / (np.linalg.norm(a) + 1e-6)
                btns[i] = (int(bot.shoot), int(bot.use_super))
                obs, _, done, _, _ = env.step(
                    {"move": bot.move.astype(np.float32),
                     "aim": bot.aim.astype(np.float32),
                     "shoot": int(bot.shoot), "super": int(bot.use_super)})
                i += 1
                if i % 5000 == 0:
                    print(f"  {i}/{n} ({i / (time.perf_counter() - t0):.0f}/s)",
                          flush=True)
        if i >= n:
            break
    np.savez_compressed(out_path, frames=frames, moves=moves, aims=aims,
                        btns=btns, ep_start=ep_start)
    print(f"saved {out_path}: {n} frames")


def merge_shards(paths: list[str], out_frames: str, out_acts: str):
    """Concatenate shard files (and optionally the original bc_data.npz +
    bc_acts_omni.npz pair) into single training files."""
    frames, moves, aims, btns, ep_start = [], [], [], [], []
    for p in paths:
        d = np.load(p)
        frames.append(d["frames"])
        ep_start.append(d["ep_start"])
        if "moves" in d:
            moves.append(d["moves"])
            aims.append(d["aims"])
            btns.append(d["btns"])
        else:  # original pair: frames in bc_data.npz, labels in ACTS_OMNI
            da = np.load(ACTS_OMNI)
            moves.append(da["moves"])
            aims.append(da["aims"])
            btns.append(da["btns"])
    np.savez_compressed(out_frames, frames=np.concatenate(frames),
                        ep_start=np.concatenate(ep_start))
    np.savez_compressed(out_acts, moves=np.concatenate(moves),
                        aims=np.concatenate(aims), btns=np.concatenate(btns))
    print(f"merged {len(frames)} shards -> {out_frames} / {out_acts} "
          f"({sum(len(f) for f in frames)} frames)")


# ----------------------------------------------------------------- training
def train(n_epochs: int, device: str, batch_size: int = 8,
          n_frames: int = 16, stride: int = 2, init: str | None = None,
          data: str = DATA, acts: str = ACTS_OMNI, tag: str = "omni",
          epoch0: int = 0):
    d = np.load(data)
    da = np.load(acts)
    frames = d["frames"]
    x_all = torch.from_numpy(frames)
    mv_all = torch.from_numpy(da["moves"])
    am_all = torch.from_numpy(da["aims"])
    bt_all = torch.from_numpy(da["btns"])
    n = len(frames)
    ep_bounds = np.flatnonzero(d["ep_start"])
    print(f"dataset: {n} frames, window={n_frames}x stride={stride} [omni]")

    net = OmniStudent(n_frames=n_frames).to(device)
    if init:
        net.load_trunk(init)
    opt = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": 1e-5},
        {"params": (list(net.osc.parameters())
                    + list(net.readout.parameters())
                    + list(net.move_head.parameters())
                    + list(net.aim_head.parameters())
                    + list(net.shoot_head.parameters())
                    + list(net.super_head.parameters())), "lr": 1e-3},
    ])
    n_val = max(1, n // 10)
    perm = np.random.RandomState(0).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    for ep in range(n_epochs):
        net.train()
        order = np.random.permutation(len(tr_idx))
        tot, nb = 0.0, 0
        t0 = time.perf_counter()
        for b in range(0, len(order), batch_size):
            idx = tr_idx[order[b:b + batch_size]]
            widx = make_windows(idx, ep_bounds, n_frames, stride)
            x = x_all[torch.from_numpy(widx)]
            x = x.permute(0, 1, 4, 2, 3).float().to(device)
            mv = mv_all[idx].clone()
            am = am_all[idx].clone()
            flip = torch.rand(len(idx)) < 0.5
            if flip.any():
                x[flip] = torch.flip(x[flip], dims=[4])
                mv[flip, 0] *= -1
                am[flip, 0] *= -1
            mv, am = mv.to(device), am.to(device)
            bt = bt_all[idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
                loss = (F.mse_loss(out["move"], mv)
                        + F.mse_loss(out["aim"], am)
                        + F.cross_entropy(out["shoot"], bt[:, 0])
                        + F.cross_entropy(out["super"], bt[:, 1]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            if nb % 200 == 0:
                print(f"  ep {ep + 1} batch {nb} loss {tot / nb:.4f} "
                      f"({nb * batch_size / (time.perf_counter() - t0):.0f} "
                      f"samples/s)", flush=True)
        # validation: stick cosine / aim angular error / button accuracy
        net.eval()
        mv_cos, am_deg, btn_ok, cnt = 0.0, 0.0, np.zeros(2), 0
        with torch.no_grad():
            for b in range(0, len(val_idx), 16):
                idx = val_idx[b:b + 16]
                widx = make_windows(idx, ep_bounds, n_frames, stride)
                x = x_all[torch.from_numpy(widx)]
                x = x.permute(0, 1, 4, 2, 3).float().to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = net(x)
                mv_t, am_t = mv_all[idx].to(device), am_all[idx].to(device)
                mv_cos += F.cosine_similarity(out["move"], mv_t).sum().item()
                pn = out["aim"] / (out["aim"].norm(dim=1, keepdim=True) + 1e-6)
                cos = (pn * am_t).sum(1).clamp(-1, 1)
                am_deg += torch.rad2deg(torch.acos(cos)).sum().item()
                btn_ok[0] += (out["shoot"].argmax(1).cpu() == bt_all[idx, 0]).sum()
                btn_ok[1] += (out["super"].argmax(1).cpu() == bt_all[idx, 1]).sum()
                cnt += len(idx)
        print(f"[epoch {ep + 1}/{n_epochs}] loss {tot / max(nb, 1):.4f} "
              f"val move_cos {mv_cos / cnt:.3f} aim_err {am_deg / cnt:.1f}deg "
              f"shoot {btn_ok[0] / cnt:.3f} super {btn_ok[1] / cnt:.3f} "
              f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
        torch.save(net.state_dict(),
                   os.path.join(OUT, f"{tag}_ep{epoch0 + ep + 1}.pt"))
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-actions", type=int, default=0)
    ap.add_argument("--collect-frames", type=int, default=0,
                    help="collect a fresh shard: frames + continuous labels")
    ap.add_argument("--seed-base", type=int, default=1042,
                    help="seed base for --collect-frames (42 = original)")
    ap.add_argument("--shard-out", type=str, default=None)
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge shard files into --data/--acts outputs")
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--init", type=str, default=None,
                    help="warm-start trunk from a discrete-head checkpoint")
    ap.add_argument("--data", type=str, default=DATA)
    ap.add_argument("--acts", type=str, default=ACTS_OMNI)
    ap.add_argument("--tag", type=str, default="omni")
    ap.add_argument("--epoch0", type=int, default=0)
    args = ap.parse_args()
    if args.collect_actions:
        collect_actions(args.collect_actions)
    if args.collect_frames:
        out = args.shard_out or os.path.join(
            OUT, f"bc_shard_s{args.seed_base}.npz")
        collect_frames(args.collect_frames, args.seed_base, out)
    if args.merge:
        merge_shards(args.merge,
                     args.data if args.data != DATA else os.path.join(
                         OUT, "bc_data_merged.npz"),
                     args.acts if args.acts != ACTS_OMNI else os.path.join(
                         OUT, "bc_acts_merged.npz"))
    if args.train:
        train(args.train, args.device, args.batch_size, init=args.init,
              data=args.data, acts=args.acts, tag=args.tag,
              epoch0=args.epoch0)


if __name__ == "__main__":
    main()
