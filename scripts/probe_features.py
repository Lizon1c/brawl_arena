"""Linear-probe audit: which representation carries fine game state?

Feature sets compared on identical frames (scripted-bot gem_grab, unit 0
driven by a bot, magenta self-ring drawn exactly as the policy sees it):
  1. frozen DINOv2 CLS (384d)          -- the current policy input
  2. frozen DINOv2 patch tokens (256x384 flattened)
  3. raw gray pixels, native 90x126 (control)
Targets: self x,y | nearest-enemy rel x,y | self hp | self ammo.
Ridge regression in DUAL (sample-space) form -> exact solution for any dim.
Split is BY GAME (no frame leakage). R^2 near 1 = linearly decodable.

Usage: .venv/Scripts/python scripts/probe_features.py [--games 8] [--every 10]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brawl_arena.core import Config, ARCHETYPES
from brawl_arena.gym_env import BrawlArenaEnv
from brawl_arena.bots import ScriptedBot
from student_rl import _lb, _self_ring

IMG = 224  # matches RLStudent.encode's interpolate target
NAMES = ["self_x", "self_y", "foe_rel_x", "foe_rel_y", "self_hp", "ammo"]


def collect(games: int, every: int, device: str, scale: int = 1):
    img_size = 224 if scale == 1 else 252   # 252 = 18*14, keeps patch grid
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                              pretrained=True).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    cls_f, patch_f, pix_f, targets, gid = [], [], [], [], []
    for g in range(games):
        env = BrawlArenaEnv(config=Config(mode="gem_grab",
                                          render_scale=6 * scale),
                            seed=31000 + g, include_frame=True)
        obs, _ = env.reset(seed=31000 + g)
        bot = ScriptedBot(seed=41000 + g)   # unit 0 plays too: real variance
        H, W = env.game.tiles.shape
        done, step = False, 0
        while not done:
            u = env.game.units[0]
            if step % every == 0 and u.alive:
                f = _lb(obs["frame"]) if scale == 1 else obs["frame"].copy()
                _self_ring(f, env.game, 0)
                t = torch.from_numpy(f).permute(2, 0, 1)[None].to(device)
                v = t.float() / 255.0
                v = (v - mean) / std
                v = F.interpolate(v, size=(img_size, img_size),
                                  mode="bilinear", align_corners=False)
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = backbone.forward_features(v)
                cls_f.append(out["x_norm_clstoken"][0].float().cpu())
                pt = out["x_norm_patchtokens"][0]
                patch_f.append(F.adaptive_avg_pool2d(
                    pt.reshape(1, img_size // 14, img_size // 14, 384)
                    .permute(0, 3, 1, 2).float(), 4).flatten().half().cpu())
                pix_f.append(F.interpolate(
                    t.float() / 255.0, (90, 126),
                    mode="bilinear", align_corners=False)[0].half().cpu())
                u = env.game.units[0]
                foes = [e for e in env.game.units[1:] if e.alive]
                if foes:
                    e = min(foes, key=lambda e: np.linalg.norm(e.pos - u.pos))
                    rel = (e.pos - u.pos) / np.array([W, H])
                else:
                    rel = np.zeros(2)
                targets.append(np.concatenate([
                    u.pos / np.array([W, H]), rel,
                    [u.hp / ARCHETYPES[u.archetype]["hp"],
                     u.ammo / 3.0]]).astype(np.float32))
                gid.append(g)
            a = bot.act(env.game, 0)
            obs, _, term, _, _ = env.step({
                "move": a.move.astype(np.float32),
                "aim": a.aim.astype(np.float32),
                "shoot": int(a.shoot), "super": int(a.use_super)})
            done, step = term, step + 1
    return (torch.stack(cls_f), torch.stack(patch_f), torch.stack(pix_f),
            torch.tensor(np.array(targets)), np.array(gid))


def ridge_r2(X, Y, gid, label):
    games = np.unique(gid)
    rng = np.random.default_rng(0)
    rng.shuffle(games)
    n_tr = max(1, int(0.8 * len(games)))
    tr, te = np.isin(gid, games[:n_tr]), np.isin(gid, games[n_tr:])
    Xtr, Xte = X[tr].float(), X[te].float()
    Ytr, Yte = Y[tr], Y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-3
    Xtr = (Xtr[:, keep] - mu[keep]) / sd[keep]
    Xte = (Xte[:, keep] - mu[keep]) / sd[keep]
    Xtr = Xtr / Xtr.shape[1] ** 0.5      # row norms ~1: lam scale-free
    Xte = Xte / Xtr.shape[1] ** 0.5
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    G = Xtr @ Xtr.T                       # dual Gram, (n_tr, n_tr)
    print(f"--- {label} (d={keep.sum()}) "
          f"train {tr.sum()} / test {te.sum()} frames ---")
    for lam in (1e-3, 1e-1, 1e1):
        alpha = torch.linalg.solve(G + lam * torch.eye(len(G)), Ytr_c)
        pred = Xte @ (Xtr.T @ alpha) + ymu
        r2 = []
        for i in range(len(NAMES)):
            ss_res = ((pred[:, i] - Yte[:, i]) ** 2).sum()
            ss_tot = ((Yte[:, i] - Yte[:, i].mean()) ** 2).sum()
            r2.append(1 - ss_res / ss_tot)
        print(f"  lam={lam:>5}: " + "  ".join(
            f"{n} {v:+.2f}" for n, v in zip(NAMES, r2)))


def mlp_r2(X, Y, gid, label, epochs=300):
    """Small nonlinear probe: fair test of what a policy head could extract."""
    games = np.unique(gid)
    rng = np.random.default_rng(0)
    rng.shuffle(games)
    n_tr = max(1, int(0.8 * len(games)))
    tr, te = np.isin(gid, games[:n_tr]), np.isin(gid, games[n_tr:])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Xte = X[tr].float().to(dev), X[te].float().to(dev)
    Ytr, Yte = Y[tr].to(dev), Y[te].to(dev)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu, ysd = Ytr.mean(0), Ytr.std(0).clamp(min=1e-6)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, Y.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Ytr_n = (Ytr - ymu) / ysd
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            loss = F.mse_loss(net(Xtr[idx]), Ytr_n[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte) * ysd + ymu
    r2 = []
    for i in range(len(NAMES)):
        ss_res = ((pred[:, i] - Yte[:, i]) ** 2).sum()
        ss_tot = ((Yte[:, i] - Yte[:, i].mean()) ** 2).sum()
        r2.append(1 - ss_res / ss_tot)
    print(f"--- MLP probe on {label} ---")
    print("       " + "  ".join(f"{n} {v:+.2f}" for n, v in zip(NAMES, r2)))


def cnn_r2(X, Y, gid, label, epochs=200):
    """Tiny conv probe: translation-equivariant readout, the proposed
    fine-stream encoder's stand-in. X: (N, 3, 90, 126) in [0, 1]."""
    games = np.unique(gid)
    rng = np.random.default_rng(0)
    rng.shuffle(games)
    n_tr = max(1, int(0.8 * len(games)))
    tr, te = np.isin(gid, games[:n_tr]), np.isin(gid, games[n_tr:])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Xte = X[tr].float().to(dev), X[te].float().to(dev)
    Ytr, Yte = Y[tr].to(dev), Y[te].to(dev)
    ymu, ysd = Ytr.mean(0), Ytr.std(0).clamp(min=1e-6)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 24, 5, 2, 2), torch.nn.ReLU(),      # 45x63
        torch.nn.Conv2d(24, 48, 3, 2, 1), torch.nn.ReLU(),     # 23x32
        torch.nn.Conv2d(48, 96, 3, 2, 1), torch.nn.ReLU(),     # 12x16
        torch.nn.Conv2d(96, 96, 3, 1, 1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(96, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, Y.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Ytr_n = (Ytr - ymu) / ysd
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            loss = F.mse_loss(net(Xtr[idx]), Ytr_n[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    r2 = []
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte), 512):
            preds.append(net(Xte[i:i + 512]))
        pred = torch.cat(preds) * ysd + ymu
    for i in range(len(NAMES)):
        ss_res = ((pred[:, i] - Yte[:, i]) ** 2).sum()
        ss_tot = ((Yte[:, i] - Yte[:, i].mean()) ** 2).sum()
        r2.append(1 - ss_res / ss_tot)
    print(f"--- CNN probe on {label} ---")
    print("       " + "  ".join(f"{n} {v:+.2f}" for n, v in zip(NAMES, r2)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--scale", type=int, default=1,
                    help="render scale multiplier: 1 = 90x126 prod, 2 = 180x252")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C, P, X, Y, gid = collect(a.games, a.every, dev, a.scale)
    print(f"collected {len(C)} frames from {a.games} games (scale x{a.scale})")
    ridge_r2(C, Y, gid, "CLS 384d ridge (n>>d, conclusive)")
    mlp_r2(C, Y, gid, "CLS 384d")
    mlp_r2(P.float(), Y, gid, "patch tokens pooled 4x4x384")
    cnn_r2(X, Y, gid, "raw RGB pixels 90x126 (tiny conv net)")
