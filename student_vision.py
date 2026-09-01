"""Student vision prototype: DINOv2-S/14 backbone + CPG-style oscillator bank.

Pipeline per user spec:
  frames -> DINOv2 ViT-S/14 (fully fine-tuned) -> per-frame 384-d feature
         -> oscillator bank (LRU-like: K damped rotation modes over the last
            T frames; population readout = the CPG-style motion encoding)
         -> 4 action heads (move 9 | aim 9 | shoot 2 | super 2)

Stages:
  --collect N   scripted bot drives unit 0 across all modes; frames resized
                to 126x126 uint8 and stored with oracle actions + episode
                boundaries (windows are gathered at train time, no dup)
  --train E     behaviour cloning on 16-frame windows (stride 2 = 32 game
                steps of context), bf16 autocast, checkpoint per epoch

Usage:
    .venv/Scripts/python student_vision.py --collect 40000
    .venv/Scripts/python student_vision.py --train 8
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
from distill_student import FLIP_LUT, split_logits

ACTION_DIMS = (9, 9, 2, 2)
IMG = 126          # DINOv2-S/14: 126/14 = 9 patches per side
DINO_DIM = 384
DATA = os.path.join("runs", "student_vision", "bc_data.npz")
OUT = os.path.join("runs", "student_vision")


# ------------------------------------------------------------------ model
class OscillatorBank(nn.Module):
    """K damped 2D rotation modes driven by the input (LRU-flavoured CPG).

    Each mode k rotates its state at a fixed learnable frequency theta_k and
    decays with learnable radius r_k in (0, 1); the motion readout is the
    population vector of all modes, mirroring CPG population coding.
    Frequencies are initialised log-spaced so the bank covers timescales
    from ~1 frame to the full window (wavelet-like scale coverage)."""

    def __init__(self, d_in: int = DINO_DIM, n_modes: int = 128):
        super().__init__()
        self.n_modes = n_modes
        self.log_r = nn.Parameter(torch.zeros(n_modes))
        t = torch.linspace(0, 1, n_modes)
        self.theta = nn.Parameter(0.05 * (2.8 / 0.05) ** t)  # 0.05..2.8 rad
        self.in_proj = nn.Linear(d_in, 2 * n_modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, 2K) population state after the last frame."""
        B, T, _ = x.shape
        K = self.n_modes
        r = torch.sigmoid(self.log_r) * 0.999
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        h = x.new_zeros(B, 2 * K)
        for t in range(T):
            u = self.in_proj(x[:, t])
            re, im = h[:, :K], h[:, K:]
            h = torch.cat([r * (re * c - im * s) + u[:, :K],
                           r * (re * s + im * c) + u[:, K:]], dim=1)
        return h


class DinoCPGStudent(nn.Module):
    """DINOv2-S/14 (fine-tuned) per frame + oscillator bank + heads."""

    def __init__(self, n_modes: int = 128, n_frames: int = 16):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2",
                                       "dinov2_vits14", pretrained=True)
        self.n_frames = n_frames
        self.osc = OscillatorBank(DINO_DIM, n_modes)
        self.readout = nn.Sequential(nn.Linear(2 * n_modes, 256), nn.ReLU())
        self.heads = nn.ModuleList(
            [nn.Linear(256, d) for d in ACTION_DIMS])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) uint8-range floats [0,255] -> (B, 22) logits."""
        B, T = x.shape[:2]
        v = x.reshape(B * T, *x.shape[2:]) / 255.0
        v = (v - self.mean) / self.std
        v = F.interpolate(v, size=(IMG, IMG), mode="bilinear",
                          align_corners=False)
        feats = self.backbone.forward_features(v)["x_norm_clstoken"]
        feats = feats.reshape(B, T, DINO_DIM)
        h = self.osc(feats)
        z = self.readout(h)
        return torch.cat([hd(z) for hd in self.heads], dim=1)


# --------------------------------------------------------------- collection
def collect(n: int, seed: int = 42):
    os.makedirs(OUT, exist_ok=True)
    frames = np.zeros((n, 90, 126, 3), dtype=np.uint8)
    acts = np.zeros((n, 4), dtype=np.int64)
    ep_start = np.zeros(n, dtype=bool)
    i = 0
    t0 = time.perf_counter()
    for round_ in range(9999):
        for mode in MODES:
            if i >= n:
                break
            env = BrawlArenaEnv(config=Config(mode=mode),
                                seed=seed + round_, include_frame=True)
            obs, _ = env.reset(seed=seed + round_)
            game = env.game
            ep_start[i] = True
            done = False
            while not done and i < n:
                bot = env.teammate_policy.act(game, 0)
                f = obs["frame"]
                if f.shape[0] != 90 or f.shape[1] != 126:
                    # showdown maps are 114x150: letterbox into 90x126
                    ff = torch.from_numpy(f).permute(2, 0, 1)[None].float()
                    ff = F.interpolate(ff, size=(90, 126), mode="bilinear",
                                       align_corners=False)
                    f = ff[0].permute(1, 2, 0).byte().numpy()
                frames[i] = f
                acts[i] = action_to_multi(bot.move, bot.aim, bot.shoot,
                                          bot.use_super)
                obs, _, terminated, _, _ = env.step(
                    {"move": bot.move.astype(np.float32),
                     "aim": bot.aim.astype(np.float32),
                     "shoot": int(bot.shoot), "super": int(bot.use_super)})
                done = terminated
                i += 1
                if i % 5000 == 0:
                    print(f"  collected {i}/{n} "
                          f"({i / (time.perf_counter() - t0):.0f}/s)")
        if i >= n:
            break
    np.savez_compressed(DATA, frames=frames, acts=acts, ep_start=ep_start)
    print(f"saved {DATA}: {len(frames)} frames")


# ----------------------------------------------------------------- training
def make_windows(idx: np.ndarray, ep_bounds: np.ndarray, n_frames: int,
                 stride: int):
    """For each sample i return frame indices [i-(T-1)*stride .. i], clamped
    below at its episode start (front-padded by repeating the first valid
    frame). ep_bounds: sorted array of episode start positions."""
    starts = ep_bounds[np.searchsorted(ep_bounds, idx, side="right") - 1]
    lags = np.arange(n_frames - 1, -1, -1) * stride       # (T,)
    j = idx[:, None] - lags[None, :]                      # (B, T)
    return np.maximum(j, starts[:, None])


def train(n_epochs: int, device: str, batch_size: int = 8,
          n_frames: int = 16, stride: int = 2):
    d = np.load(DATA)
    frames, acts, ep_start = d["frames"], d["acts"], d["ep_start"]
    x_all = torch.from_numpy(frames)            # uint8 (N, 90, 126, 3), cpu
    y_all = torch.from_numpy(acts)
    n = len(frames)
    ep_bounds = np.flatnonzero(ep_start)
    print(f"dataset: {n} frames, window={n_frames}x stride={stride}")

    net = DinoCPGStudent(n_frames=n_frames).to(device)
    # differential lr: backbone gentle, new modules aggressive
    opt = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": 1e-5},
        {"params": (list(net.osc.parameters())
                    + list(net.readout.parameters())
                    + list(net.heads.parameters())), "lr": 1e-3},
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
            x = x_all[torch.from_numpy(widx)]          # (B,T,90,126,3)
            x = x.permute(0, 1, 4, 2, 3).float().to(device)
            y = y_all[idx].clone()
            flip = torch.rand(len(idx)) < 0.5
            if flip.any():
                x[flip] = torch.flip(x[flip], dims=[4])
                y[flip, 0] = FLIP_LUT[y[flip, 0]]
                y[flip, 1] = FLIP_LUT[y[flip, 1]]
            y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
                loss = sum(F.cross_entropy(lg, y[:, k])
                           for k, lg in enumerate(split_logits(logits)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            if nb % 200 == 0:
                print(f"  ep {ep + 1} batch {nb} loss {tot / nb:.4f} "
                      f"({nb * batch_size / (time.perf_counter() - t0):.0f} "
                      f"samples/s)")
        # validation per head
        net.eval()
        correct = np.zeros(4)
        with torch.no_grad():
            for b in range(0, len(val_idx), 16):
                idx = val_idx[b:b + 16]
                widx = make_windows(idx, ep_bounds, n_frames, stride)
                x = x_all[torch.from_numpy(widx)]
                x = x.permute(0, 1, 4, 2, 3).float().to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(x)
                for k, lg in enumerate(split_logits(pred)):
                    correct[k] += (lg.argmax(1).cpu()
                                   == y_all[idx, k]).sum()
        acc = correct / len(val_idx)
        print(f"[epoch {ep + 1}/{n_epochs}] loss {tot / max(nb, 1):.4f} "
              f"val acc move {acc[0]:.3f} aim {acc[1]:.3f} "
              f"shoot {acc[2]:.3f} super {acc[3]:.3f} "
              f"({(time.perf_counter() - t0) / 60:.1f} min)")
        torch.save(net.state_dict(),
                   os.path.join(OUT, f"dino_cpg_ep{ep + 1}.pt"))
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, default=0)
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    if args.collect:
        collect(args.collect)
    if args.train:
        train(args.train, args.device, args.batch_size)


if __name__ == "__main__":
    main()
