"""Stage-1 CNN perception pretrain (v5): grow the visual cortex from scratch.

Replaces the frozen DINOv2 front-end (probe-audited 09-04: enemy pos / hp /
ammo are undecodable from its features, R2 < 0) with a small trainable CNN
(probe-verified: self pos .94, foe rel .4-.5, hp .99, ammo .85).

Data: fresh scripted-bot games across all modes. Frames carry the magenta
self-ring + fixed self panel (EXACTLY the rollout-time interface, applied via
student_rl._self_ring) — the old BC shards predate the ring and the new
render, so they are not reusable. Labels per frame, free from the sim:
  - actions (move / aim / shoot / super) from the scripted bot driving unit 0
  - detection grid 12x16 x 6: [presence, team, dx, dy, hp_frac, item]
    (units hidden in bushes from viewer team 0 are NOT labelled: not drawn,
    not detectable)

Losses: action BC on the pooled 384-d feature + detection on the spatial map.
The trunk (RoPE transformer) is untouched here: stage 2 reuses
student_tx_pretrain with --backbone cnn (CNN frozen, features cached exactly
like the DINO era).

Usage:
    .venv/Scripts/python student_cnn_pretrain.py --collect 60000
    .venv/Scripts/python student_cnn_pretrain.py --train 10
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import ARCHETYPES, MODES
from student_vision import CNNEncoder, DINO_DIM
from student_rl import _self_ring

OUT = os.path.join("runs", "student_cnn")
DATA = os.path.join(OUT, "cnn_data.npz")
GH, GW = 12, 16          # detection grid over the 90x126 frame
DET_C = 6                # presence, team, dx, dy, hp_frac, item


# ------------------------------------------------------------ label building
def det_target(game) -> np.ndarray:
    """(GH, GW, 6) float32 detection target from ground-truth game state."""
    t = np.zeros((GH, GW, DET_C), dtype=np.float32)
    mw, mh = game.cfg.map_w, game.cfg.map_h
    for u in game.units:
        if not u.alive:
            continue
        if u.team != 0 and not game.is_visible_to(u, 0):
            continue                      # stealthed in a bush: not drawn
        nx, ny = u.pos[0] / mw, u.pos[1] / mh
        gx, gy = nx * GW, ny * GH
        cx, cy = min(int(gx), GW - 1), min(int(gy), GH - 1)
        t[cy, cx, 0] = 1.0
        t[cy, cx, 1] = float(u.team)
        t[cy, cx, 2] = gx - cx - 0.5      # offset within cell, [-.5, .5]
        t[cy, cx, 3] = gy - cy - 0.5
        t[cy, cx, 4] = max(0.0, u.hp / ARCHETYPES[u.archetype]["hp"])
    pts = [g for g in game.gems]
    if game.mode == "brawl_ball" and game.ball_carrier is None:
        pts.append(game.ball_pos)
    for p in pts:
        cx = min(int(p[0] / mw * GW), GW - 1)
        cy = min(int(p[1] / mh * GH), GH - 1)
        t[cy, cx, 5] = 1.0
    return t


# --------------------------------------------------------------- collection
def collect(n: int, out_path: str, seed_base: int = 42):
    os.makedirs(OUT, exist_ok=True)
    frames = np.zeros((n, 90, 126, 3), dtype=np.uint8)
    moves = np.zeros((n, 2), dtype=np.float32)
    aims = np.zeros((n, 2), dtype=np.float32)
    btns = np.zeros((n, 2), dtype=np.int64)
    dets = np.zeros((n, GH, GW, DET_C), dtype=np.float32)
    ep_start = np.zeros(n, dtype=bool)
    i = 0
    t0 = time.perf_counter()
    for round_ in range(99999):
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
                f = f.copy()
                _self_ring(f, game, 0)    # ring + self panel: the rollout IF
                frames[i] = f
                moves[i] = bot.move
                a = np.asarray(bot.aim, dtype=np.float32)
                aims[i] = a / (np.linalg.norm(a) + 1e-6)
                btns[i] = (int(bot.shoot), int(bot.use_super))
                dets[i] = det_target(game)
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
                        btns=btns, dets=dets, ep_start=ep_start)
    print(f"saved {out_path}: {n} frames")


# ------------------------------------------------------------------ training
def det_loss(pred: torch.Tensor, tgt: torch.Tensor):
    """pred (B, 6, GH, GW); tgt (B, GH, GW, 6)."""
    p = pred.permute(0, 2, 3, 1)
    occ = tgt[..., 0] > 0
    l_pres = F.binary_cross_entropy_with_logits(
        p[..., 0], tgt[..., 0], pos_weight=torch.tensor(10.0, device=p.device))
    l_item = F.binary_cross_entropy_with_logits(
        p[..., 5], tgt[..., 5], pos_weight=torch.tensor(5.0, device=p.device))
    if occ.any():
        l_reg = (F.mse_loss(torch.sigmoid(p[..., 1])[occ], tgt[..., 1][occ])
                 + F.mse_loss(torch.tanh(p[..., 2:4])[occ], tgt[..., 2:4][occ])
                 + F.mse_loss(torch.sigmoid(p[..., 4])[occ], tgt[..., 4][occ]))
    else:
        l_reg = p.sum() * 0.0
    return l_pres + l_item + l_reg


def train(n_epochs: int, device: str, batch_size: int = 256, lr: float = 1e-3,
          data: str = DATA, out: str = os.path.join(OUT, "cnn_stage1.pt"),
          det_w: float = 1.0, act_w: float = 1.0):
    d = np.load(data)
    x_all = torch.from_numpy(d["frames"])          # (N, 90, 126, 3) uint8
    mv_all = torch.from_numpy(d["moves"])
    am_all = torch.from_numpy(d["aims"])
    bt_all = torch.from_numpy(d["btns"])
    det_all = torch.from_numpy(d["dets"])          # (N, GH, GW, 6)
    ep_id = np.cumsum(d["ep_start"]) - 1           # episode per frame
    n = len(x_all)
    eps = np.unique(ep_id)
    rng = np.random.RandomState(0)
    rng.shuffle(eps)
    n_val_ep = max(1, len(eps) // 10)
    val_mask = np.isin(ep_id, eps[:n_val_ep])
    val_idx, tr_idx = np.flatnonzero(val_mask), np.flatnonzero(~val_mask)
    print(f"dataset: {n} frames ({len(eps)} episodes), "
          f"train {len(tr_idx)} / val {len(val_idx)} [split by episode]")

    net = CNNEncoder(DINO_DIM).to(device)
    move_h = torch.nn.Linear(DINO_DIM, 2).to(device)
    aim_h = torch.nn.Linear(DINO_DIM, 2).to(device)
    shoot_h = torch.nn.Linear(DINO_DIM, 2).to(device)
    super_h = torch.nn.Linear(DINO_DIM, 2).to(device)
    params = (list(net.parameters()) + list(move_h.parameters())
              + list(aim_h.parameters()) + list(shoot_h.parameters())
              + list(super_h.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    steps_per_ep = (len(tr_idx) + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * steps_per_ep, eta_min=lr / 10)

    def batch_loss(idx: np.ndarray, flip: bool):
        x = x_all[idx].permute(0, 3, 1, 2).to(device)        # (B,3,90,126)
        mv, am = mv_all[idx].to(device), am_all[idx].to(device)
        bt = bt_all[idx].to(device)
        dt = det_all[idx].to(device)
        if flip:                       # maps are mirror-symmetric: free x2
            x = torch.flip(x, dims=[3])
            mv = mv * torch.tensor([-1.0, 1.0], device=device)
            am = am * torch.tensor([-1.0, 1.0], device=device)
            dt = torch.flip(dt, dims=[2])
            dt[..., 2] = -dt[..., 2]
        feat, det = net(x)
        l_det = det_loss(det, dt)
        l_act = (F.mse_loss(torch.tanh(move_h(feat)), mv)
                 + F.mse_loss(aim_h(feat), am)
                 + F.cross_entropy(shoot_h(feat), bt[:, 0])
                 + F.cross_entropy(super_h(feat), bt[:, 1]))
        return det_w * l_det + act_w * l_act, l_det, l_act

    for ep in range(n_epochs):
        net.train()
        order = np.random.permutation(len(tr_idx))
        tot = td = ta = 0.0
        nb = 0
        t0 = time.perf_counter()
        for b in range(0, len(order), batch_size):
            idx = tr_idx[order[b:b + batch_size]]
            loss, ld, la = batch_loss(idx, flip=bool(np.random.rand() < 0.5))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            sched.step()
            tot += float(loss); td += float(ld); ta += float(la)
            nb += 1
        net.eval()
        vt = vd = va = 0.0
        vnb = 0
        with torch.no_grad():
            for b in range(0, len(val_idx), batch_size):
                loss, ld, la = batch_loss(val_idx[b:b + batch_size],
                                          flip=False)
                vt += float(loss); vd += float(ld); va += float(la)
                vnb += 1
        print(f"[epoch {ep + 1}/{n_epochs}] loss {tot / max(nb, 1):.4f} "
              f"(det {td / max(nb, 1):.4f} act {ta / max(nb, 1):.4f}) "
              f"val {vt / max(vnb, 1):.4f} "
              f"(det {vd / max(vnb, 1):.4f} act {va / max(vnb, 1):.4f}) "
              f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
        os.makedirs(OUT, exist_ok=True)
        torch.save({"cnn": net.state_dict()}, out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, default=0)
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--data", type=str, default=DATA)
    ap.add_argument("--out", type=str, default=os.path.join(OUT,
                                                           "cnn_stage1.pt"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    if args.collect:
        collect(args.collect, args.data)
    if args.train:
        train(args.train, args.device, args.batch_size, args.lr,
              args.data, args.out)


if __name__ == "__main__":
    main()
